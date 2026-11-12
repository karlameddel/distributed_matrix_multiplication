import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.map.IMap;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;

/**
 * One distributed task: computes one strip of bs rows of C = A * B on a Hazelcast member.
 *
 * Partitioning (1-D, by rows):
 *   - A: the task carries its own strip of A (rows [strip*bs, min((strip+1)*bs, n))), so A is
 *     scattered to the members together with the tasks.
 *   - B: every strip needs all of B. The client stores B in the IMap MAP as row strips
 *     key(job, "B", s). Each member reads B once per job and keeps it in this JVM, so B crosses
 *     the cluster once per member, not once per task.
 *   - C: the task returns its strip of C to the client inside a Result.
 *
 * The kernel is the same as ParallelBlocked.strip (same tiles, i-k-j order, float64), with the rows
 * renumbered from 0. For every C[i][j] the products are added in ascending k, so the result is
 * expected to be bit-identical to Blocked.
 */
public final class StripTask implements Callable<StripTask.Result>, Serializable, HazelcastInstanceAware {
    private static final long serialVersionUID = 1L;

    public static final String MAP = "matrices";

    private final String job;   // unique per multiplication, so the B cache never mixes two runs
    private final int strip, n, bs;
    private final double[] a;   // this strip of A
    private transient HazelcastInstance hz;

    public StripTask(String job, int strip, int n, int bs, double[] a) {
        this.job = job; this.strip = strip; this.n = n; this.bs = bs; this.a = a;
    }

    public static String key(String job, String matrix, int strip) {
        return job + ":" + matrix + ":" + strip;
    }

    @Override
    public void setHazelcastInstance(HazelcastInstance hz) {   // called by Hazelcast on the member
        this.hz = hz;
    }

    @Override
    public Result call() {
        long t0 = System.nanoTime();
        boolean[] fetched = new boolean[1];
        double[] b = fullB(hz.<String, double[]>getMap(MAP), job, n, bs, fetched);
        long t1 = System.nanoTime();
        double[] c = multiplyRows(a, b, n, bs);
        long t2 = System.nanoTime();
        String member = hz.getCluster().getLocalMember().getAddress().toString();
        return new Result(strip, c, member, Thread.currentThread().getName(), fetched[0], t1 - t0, t2 - t1);
    }

    /** What a task sends back: its strip of C and where / how long it ran (member-side clock). */
    public static final class Result implements Serializable {
        private static final long serialVersionUID = 1L;
        public final int strip;
        public final double[] c;
        public final String member, thread;
        public final boolean fetchedB;   // true for the one task per member and job that read B from the IMap
        public final long bNs;           // time to obtain B (read it, or wait for the task that reads it)
        public final long computeNs;     // multiplyRows only

        Result(int strip, double[] c, String member, String thread, boolean fetchedB, long bNs, long computeNs) {
            this.strip = strip; this.c = c; this.member = member; this.thread = thread;
            this.fetchedB = fetchedB; this.bNs = bNs; this.computeNs = computeNs;
        }
    }

    // ---- B cache: one copy per member JVM and per job --------------------------------------------

    private static String cachedJob;
    private static double[] cachedB;

    private static synchronized double[] fullB(IMap<String, double[]> map, String job, int n, int bs, boolean[] fetched) {
        if (!job.equals(cachedJob)) {
            cachedB = null;                                   // free the previous job's B first
            int strips = (n + bs - 1) / bs;
            Set<String> keys = new HashSet<>();
            for (int s = 0; s < strips; s++) keys.add(key(job, "B", s));
            Map<String, double[]> parts = map.getAll(keys);   // one bulk read
            double[] b = new double[n * n];
            for (int s = 0; s < strips; s++) {
                double[] part = parts.get(key(job, "B", s));
                if (part == null) throw new IllegalStateException("missing " + key(job, "B", s));
                System.arraycopy(part, 0, b, s * bs * n, part.length);
            }
            cachedB = b;
            cachedJob = job;
            fetched[0] = true;
        }
        return cachedB;
    }

    // ---- kernel ----------------------------------------------------------------------------------

    /** C_strip = A_strip * B, where A_strip has a.length / n rows. Same loops as ParallelBlocked.strip. */
    public static double[] multiplyRows(double[] a, double[] b, int n, int bs) {
        int rows = a.length / n;
        double[] c = new double[rows * n];                    // zero-initialised by Java
        for (int kk = 0; kk < n; kk += bs) {
            int kMax = Math.min(kk + bs, n);
            for (int jj = 0; jj < n; jj += bs) {
                int jMax = Math.min(jj + bs, n);
                for (int i = 0; i < rows; i++) {
                    int co = i * n;
                    for (int k = kk; k < kMax; k++) {
                        double aik = a[i * n + k];
                        int bo = k * n;
                        for (int j = jj; j < jMax; j++)
                            c[co + j] += aik * b[bo + j];
                    }
                }
            }
        }
        return c;
    }
}