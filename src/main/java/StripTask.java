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
 * Data in the IMap MAP (written by the client before the tasks are submitted):
 *   key(job, "A", s) -> rows [s*bs, min((s+1)*bs, n)) of A, row-major
 *   key(job, "B", s) -> the same rows of B
 * The task is meant to be submitted with submitToKeyOwner(task, key(job, "A", s)), so it runs on
 * the member that owns its strip of A and reading A is local. B is needed in full: it is read once
 * per member and per job (cached in this JVM), so it crosses the cluster once per member, not once per task.
 *
 * The kernel is the same as ParallelBlocked.strip (same tiles, i-k-j order, float64), with the rows
 * renumbered from 0. For every C[i][j] the products are added in ascending k, so the result is
 * expected to be bit-identical to Blocked.
 */
public final class StripTask implements Callable<double[]>, Serializable, HazelcastInstanceAware {
    private static final long serialVersionUID = 1L;

    public static final String MAP = "matrices";

    private final String job;   // unique per multiplication, so the B cache never mixes two runs
    private final int strip, n, bs;
    private transient HazelcastInstance hz;

    public StripTask(String job, int strip, int n, int bs) {
        this.job = job; this.strip = strip; this.n = n; this.bs = bs;
    }

    public static String key(String job, String matrix, int strip) {
        return job + ":" + matrix + ":" + strip;
    }

    @Override
    public void setHazelcastInstance(HazelcastInstance hz) {   // called by Hazelcast on the member
        this.hz = hz;
    }

    @Override
    public double[] call() {
        IMap<String, double[]> map = hz.getMap(MAP);
        double[] a = map.get(key(job, "A", strip));
        if (a == null) throw new IllegalStateException("missing " + key(job, "A", strip));
        double[] b = fullB(map, job, n, bs);
        return multiplyRows(a, b, n, bs);
    }

    // ---- B cache: one copy per member JVM and per job --------------------------------------------

    private static String cachedJob;
    private static double[] cachedB;

    private static synchronized double[] fullB(IMap<String, double[]> map, String job, int n, int bs) {
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