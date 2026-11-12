import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.cluster.Member;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IExecutorService;
import com.hazelcast.map.IMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.Future;

/**
 * Client side of the distributed blocked multiplication (Hazelcast).
 *
 * The client is a HazelcastClient, not a member: it stores no data and runs no tasks, so the
 * workers are exactly the WorkerNode members.
 *
 * One multiplication = 4 consecutive phases, each timed on the client with System.nanoTime():
 *   1. upload   : B is written to the IMap as row strips of bs rows.
 *   2. execute  : one StripTask per strip of bs rows, assigned round-robin to the members
 *                 (strip s -> member s mod p). Each task carries its strip of A. Ends when the
 *                 last strip of C has arrived. Includes sending A, reading B on the members,
 *                 the computation and returning C; these overlap and are NOT separable here
 *                 (the member-side bNs / computeNs in the Results describe them).
 *   3. assemble : the strips of C are copied into C.
 *   4. cleanup  : the IMap is cleared.
 * End-to-end time = upload + execute + assemble. Cleanup is reported separately.
 */
public final class DistributedBlocked {
    private DistributedBlocked() {}

    public static final String EXECUTOR = "matmul";

    /** Client-side phase times and the member-side summary of one multiplication. */
    public static final class Timing {
        public long uploadNs, executeNs, assembleNs, cleanupNs;
        public int members, tasks, bFetches;
        public long sumComputeNs, maxComputeNs, maxBNs;
        public final Map<String, Integer> tasksPerMember = new TreeMap<>();

        public long endToEndNs() { return uploadNs + executeNs + assembleNs; }
    }

    /** Connects a client to the local cluster started with config/hazelcast.xml. */
    public static HazelcastInstance connect(String clusterName, String address) {
        ClientConfig cc = new ClientConfig();
        cc.setClusterName(clusterName);
        cc.getNetworkConfig().addAddress(address);
        cc.setProperty("hazelcast.phone.home.enabled", "false");
        return HazelcastClient.newHazelcastClient(cc);
    }

    public static Timing multiply(HazelcastInstance hz, double[] A, double[] B, double[] C, int n, int bs)
            throws Exception {
        String job = UUID.randomUUID().toString();
        IMap<String, double[]> map = hz.getMap(StripTask.MAP);
        IExecutorService executor = hz.getExecutorService(EXECUTOR);
        List<Member> members = new ArrayList<>(hz.getCluster().getMembers());
        if (members.isEmpty()) throw new IllegalStateException("no members in the cluster");
        int strips = (n + bs - 1) / bs;
        Timing t = new Timing();
        t.members = members.size();
        t.tasks = strips;

        // 1. upload B
        long t0 = System.nanoTime();
        for (int s = 0; s < strips; s++) {
            int i0 = s * bs, i1 = Math.min(i0 + bs, n);
            map.set(StripTask.key(job, "B", s), Arrays.copyOfRange(B, i0 * n, i1 * n));
        }
        long t1 = System.nanoTime();

        // 2. execute: scatter A with the tasks, gather the strips of C
        List<Future<StripTask.Result>> futures = new ArrayList<>(strips);
        for (int s = 0; s < strips; s++) {
            int i0 = s * bs, i1 = Math.min(i0 + bs, n);
            StripTask task = new StripTask(job, s, n, bs, Arrays.copyOfRange(A, i0 * n, i1 * n));
            futures.add(executor.submitToMember(task, members.get(s % members.size())));
        }
        List<StripTask.Result> results = new ArrayList<>(strips);
        for (Future<StripTask.Result> f : futures) results.add(f.get());
        long t2 = System.nanoTime();

        // 3. assemble C
        for (StripTask.Result r : results)
            System.arraycopy(r.c, 0, C, r.strip * bs * n, r.c.length);
        long t3 = System.nanoTime();

        // 4. cleanup
        map.clear();
        long t4 = System.nanoTime();

        t.uploadNs = t1 - t0; t.executeNs = t2 - t1; t.assembleNs = t3 - t2; t.cleanupNs = t4 - t3;
        for (StripTask.Result r : results) {
            t.tasksPerMember.merge(r.member, 1, Integer::sum);
            if (r.fetchedB) t.bFetches++;
            t.sumComputeNs += r.computeNs;
            t.maxComputeNs = Math.max(t.maxComputeNs, r.computeNs);
            t.maxBNs = Math.max(t.maxBNs, r.bNs);
        }
        return t;
    }

    /**
     * Smoke test (start the members first):
     *   java -cp target/dmm.jar DistributedBlocked [n] [block]
     * Multiplies random n x n matrices on the cluster and compares C with Blocked.
     */
    public static void main(String[] args) throws Exception {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 300;
        int bs = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        Random rnd = new Random(42);
        double[] A = new double[n * n], B = new double[n * n], C = new double[n * n], ref = new double[n * n];
        for (int i = 0; i < n * n; i++) { A[i] = rnd.nextDouble(); B[i] = rnd.nextDouble(); }
        Blocked.multiply(A, B, ref, n, bs);

        HazelcastInstance hz = connect("matmul", "127.0.0.1:5701");
        try {
            Timing t = multiply(hz, A, B, C, n, bs);
            boolean same = Arrays.equals(C, ref);
            System.out.println("#n=" + n + " block=" + bs + " members=" + t.members + " tasks=" + t.tasks);
            System.out.println("#tasks_per_member=" + t.tasksPerMember);
            System.out.printf("#upload_ms=%.3f execute_ms=%.3f assemble_ms=%.3f cleanup_ms=%.3f end_to_end_ms=%.3f%n",
                    t.uploadNs / 1e6, t.executeNs / 1e6, t.assembleNs / 1e6, t.cleanupNs / 1e6, t.endToEndNs() / 1e6);
            System.out.printf("#member_side: b_fetches=%d max_b_ms=%.3f sum_compute_ms=%.3f max_compute_ms=%.3f%n",
                    t.bFetches, t.maxBNs / 1e6, t.sumComputeNs / 1e6, t.maxComputeNs / 1e6);
            System.out.println(same ? "OK: bit-identical to Blocked" : "FAIL: result differs from Blocked");
            if (!same) System.exit(1);
        } finally {
            hz.shutdown();
        }
    }
}