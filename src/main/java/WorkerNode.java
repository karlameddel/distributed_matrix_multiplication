import com.hazelcast.config.Config;
import com.hazelcast.config.FileSystemXmlConfig;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

/**
 * Starts one Hazelcast member (one JVM process) and keeps it running until the process is stopped.
 * Run it p times to obtain a cluster of p members:
 *   java -cp target/dmm.jar WorkerNode [config/hazelcast.xml] [executorThreads]
 *
 * executorThreads (default 1) is the number of threads of this member that run StripTasks,
 * i.e. how many strips one member computes at the same time.
 */
public final class WorkerNode {
    private WorkerNode() {}

    public static void main(String[] args) throws Exception {
        String configPath = args.length > 0 ? args[0] : "config/hazelcast.xml";
        int threads = args.length > 1 ? Integer.parseInt(args[1]) : 1;
        if (threads < 1) throw new IllegalArgumentException("executorThreads must be >= 1");

        Config config = new FileSystemXmlConfig(configPath);
        config.getExecutorConfig(DistributedBlocked.EXECUTOR).setPoolSize(threads);
        HazelcastInstance hz = Hazelcast.newHazelcastInstance(config);

        System.out.println("#member=" + hz.getCluster().getLocalMember().getAddress());
        System.out.println("#cluster_size=" + hz.getCluster().getMembers().size());
        System.out.println("#executor_threads=" + threads);
        // Hazelcast's threads keep the JVM alive; Ctrl+C stops the member.
    }
}