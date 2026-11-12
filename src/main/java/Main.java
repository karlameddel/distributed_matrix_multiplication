import com.hazelcast.core.HazelcastInstance;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Assignment 4 benchmark driver (one configuration per process).
 *
 * Usage:
 *   java -Xmx4g -cp target/dmm.jar Main --method seq|par|dist --a A.bin --b B.bin --n N
 *        [--threads 1] [--block 64] [--reps 5] [--warmup 3] [--warmup-min-ms 300]
 *        [--batch 0] [--target-ms 20] [--out C.bin]
 *        [--members P] [--cluster matmul] [--address 127.0.0.1:5701]      (dist only)
 *
 *   seq   Blocked (sequential baseline of Assignments 2 and 3, unchanged). Runs on the main thread.
 *   par   ParallelBlocked (Assignment 3) with a fixed pool of exactly --threads worker threads.
 *   dist  DistributedBlocked: this process is a Hazelcast CLIENT of an already running cluster of
 *         --members WorkerNode members (started by the harness). The run aborts if the cluster
 *         does not have exactly --members members.
 *
 * Timed region
 *   seq / par : the multiplication only (as in Assignment 3).
 *   dist      : end-to-end = upload of B + execute (send A, tasks, return of C) + assembly of C.
 *               The phases are also reported separately. Cleanup of the IMap is outside and is
 *               reported on its own. Connecting the client is outside (setup, see #connect_s).
 * Input loading, the allocation of C and the checksum are outside the timer for every method.
 *
 * Warm-up: at least --warmup repetitions AND at least --warmup-min-ms, excluded from the samples.
 * Batching: --batch 0 (auto) sizes each sample to about --target-ms for seq / par. dist always uses
 * batch = 1 (one multiplication per sample) so that its phases belong to a single multiplication.
 * Reported times are always seconds PER multiplication.
 *
 * stdout: one CSV line per measured sample
 *   rep,batch,seconds_per_op,checksum,upload_s,execute_s,assemble_s,cleanup_s,
 *   b_fetches,max_b_s,sum_compute_s,max_compute_s
 * (the last 8 columns are empty for seq / par), followed by "#key=value" metadata lines.
 */
public final class Main {
    private static double[] A, B, C;

    public static void main(String[] args) throws Exception {
        Map<String, String> o = parse(args);
        String method = req(o, "method");
        int n = Integer.parseInt(req(o, "n"));
        int block = Integer.parseInt(o.getOrDefault("block", "64"));
        int threads = Integer.parseInt(o.getOrDefault("threads", "1"));
        int reps = Integer.parseInt(o.getOrDefault("reps", "5"));
        int warmup = Integer.parseInt(o.getOrDefault("warmup", "3"));
        long warmupMinMs = Long.parseLong(o.getOrDefault("warmup-min-ms", "300"));
        int batchOpt = Integer.parseInt(o.getOrDefault("batch", "0"));
        double targetMs = Double.parseDouble(o.getOrDefault("target-ms", "20"));
        if (!method.equals("seq") && !method.equals("par") && !method.equals("dist"))
            throw new IllegalArgumentException("unknown method: " + method);
        if (!method.equals("par") && threads != 1)
            throw new IllegalArgumentException("--threads is only for par (dist: threads per member are set on WorkerNode)");
        if (block < 1 || threads < 1) throw new IllegalArgumentException("block and threads must be >= 1");

        A = MatrixIO.load(req(o, "a"), n);
        B = MatrixIO.load(req(o, "b"), n);
        C = new double[n * n];

        if (method.equals("dist")) runDist(o, n, block, reps, warmup, warmupMinMs);
        else runLocal(method, n, block, threads, reps, warmup, warmupMinMs, batchOpt, targetMs);

        if (o.containsKey("out")) MatrixIO.save(o.get("out"), C, n);
        java.lang.ref.Reference.reachabilityFence(A);
        java.lang.ref.Reference.reachabilityFence(B);
        java.lang.ref.Reference.reachabilityFence(C);
    }

    // ---- seq / par: unchanged protocol of Assignment 3 --------------------------------------------

    private static void runLocal(String method, int n, int block, int threads, int reps, int warmup,
                                 long warmupMinMs, int batchOpt, double targetMs) {
        ExecutorService pool = method.equals("par") ? Executors.newFixedThreadPool(threads) : null;
        Runnable op = method.equals("seq")
                ? () -> Blocked.multiply(A, B, C, n, block)
                : () -> ParallelBlocked.multiply(A, B, C, n, block, pool);

        long wStart = System.nanoTime();
        int w = 0;
        double lastNs = 0;
        while (w < 100000 && (w < warmup || (System.nanoTime() - wStart) / 1e6 < warmupMinMs)) {
            long t0 = System.nanoTime();
            op.run();
            lastNs = System.nanoTime() - t0;
            w++;
        }

        int batch;
        if (batchOpt > 0) batch = batchOpt;
        else if (w == 0) batch = 1;
        else batch = (int) Math.max(1, Math.min(100000, Math.ceil(targetMs * 1e6 / Math.max(1.0, lastNs))));

        for (int r = 0; r < reps; r++) {
            long t0 = System.nanoTime();
            for (int b = 0; b < batch; b++) op.run();
            long t1 = System.nanoTime();
            double secPerOp = (t1 - t0) / 1e9 / batch;
            double checksum = Checksum.sum(C);
            System.out.printf(Locale.ROOT, "%d,%d,%.9e,%.12e,,,,,,,,%n", r, batch, secPerOp, checksum);
            System.out.flush();
        }

        int jvmThreads = Thread.activeCount();
        if (pool != null) pool.shutdown();

        System.out.println("#threads=" + (pool == null ? 1 : threads));
        System.out.println("#tasks_per_multiplication=" + (pool == null ? 1 : (n + block - 1) / block));
        System.out.println("#active_java_threads=" + jvmThreads);
        System.out.println("#warmup_iterations=" + w);
        printCommonMetadata(n);
    }

    // ---- dist -------------------------------------------------------------------------------------

    private static void runDist(Map<String, String> o, int n, int block, int reps, int warmup, long warmupMinMs)
            throws Exception {
        int expected = Integer.parseInt(req(o, "members"));
        String cluster = o.getOrDefault("cluster", "matmul");
        String address = o.getOrDefault("address", "127.0.0.1:5701");

        long c0 = System.nanoTime();
        HazelcastInstance hz = DistributedBlocked.connect(cluster, address);
        long c1 = System.nanoTime();
        try {
            int members = hz.getCluster().getMembers().size();
            if (members != expected)
                throw new IllegalStateException("cluster has " + members + " members, expected " + expected);

            long wStart = System.nanoTime();
            int w = 0;
            while (w < 100000 && (w < warmup || (System.nanoTime() - wStart) / 1e6 < warmupMinMs)) {
                DistributedBlocked.multiply(hz, A, B, C, n, block);
                w++;
            }

            DistributedBlocked.Timing last = null;
            for (int r = 0; r < reps; r++) {
                DistributedBlocked.Timing t = DistributedBlocked.multiply(hz, A, B, C, n, block);
                double checksum = Checksum.sum(C);
                System.out.printf(Locale.ROOT, "%d,1,%.9e,%.12e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e%n",
                        r, t.endToEndNs() / 1e9, checksum,
                        t.uploadNs / 1e9, t.executeNs / 1e9, t.assembleNs / 1e9, t.cleanupNs / 1e9,
                        t.bFetches, t.maxBNs / 1e9, t.sumComputeNs / 1e9, t.maxComputeNs / 1e9);
                System.out.flush();
                last = t;
            }

            System.out.println("#members=" + members);
            System.out.println("#tasks_per_multiplication=" + (n + block - 1) / block);
            if (last != null) System.out.println("#tasks_per_member=" + last.tasksPerMember);
            System.out.printf(Locale.ROOT, "#connect_s=%.6f%n", (c1 - c0) / 1e9);
            System.out.println("#active_java_threads=" + Thread.activeCount());
            System.out.println("#warmup_iterations=" + w);
            printCommonMetadata(n);
        } finally {
            hz.shutdown();
        }
    }

    // ---- helpers ----------------------------------------------------------------------------------

    private static void printCommonMetadata(int n) {
        System.gc();
        Runtime rt = Runtime.getRuntime();
        long live = rt.totalMemory() - rt.freeMemory();
        System.out.println("#live_heap_bytes_after_gc=" + live);
        System.out.println("#array_estimate_bytes=" + 24L * n * n);
        System.out.println("#java_version=" + System.getProperty("java.version") + " " + System.getProperty("java.vm.name"));
        System.out.println("#available_processors=" + rt.availableProcessors());
        System.out.println("#max_heap_bytes=" + rt.maxMemory());
    }

    private static String req(Map<String, String> o, String k) {
        String v = o.get(k);
        if (v == null) throw new IllegalArgumentException("missing --" + k);
        return v;
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (!args[i].startsWith("--") || i + 1 >= args.length)
                throw new IllegalArgumentException("bad argument near '" + args[i] + "'");
            m.put(args[i].substring(2), args[i + 1]);
        }
        return m;
    }
}