# Distributed Matrix Multiplication

**Big Data · Assignment 4 · Academic year 2026-2027**

## Research question

> How do the runtime, speedup, efficiency and memory of a blocked matrix multiplication distributed
> over Hazelcast members change with the number of members and the matrix size, compared with the
> sequential and shared-memory parallel versions of Assignment 3, and how much of the end-to-end time
> is computation and how much is data transfer and framework overhead?

Conclusions are limited to the tested implementation, inputs and hardware. **All members and the
client run on one laptop** (localhost, no physical network): the results measure the cost of
distributing the work over separate JVM processes with Hazelcast, not the behaviour of a real
multi-machine cluster. Weak scaling was not studied.

## Reuse from Assignment 3

| Reused | Where | Note |
|---|---|---|
| Sequential baseline (blocked, bs = 64) | `src/main/java/Blocked.java` | Unchanged kernel of Assignments 2 and 3 |
| Shared-memory parallel version | `src/main/java/ParallelBlocked.java` | Unchanged (Assignment 3) |
| Matrix I/O and checksum | `src/main/java/MatrixIO.java`, `Checksum.java` | Unchanged |
| Input matrices | `scripts/generate_inputs.py` | Same seeds `[42, n, tag]` -> same bytes; `--check-a3` compares SHA-256 with the Assignment 3 manifest |
| Harness protocol | `scripts/run_benchmarks.py` | One client process per configuration, probe + timing run, limits 300 s / 4 GB, median and IQR |
| Correctness criterion | `scripts/verify_correctness.py` | `rtol = 1e-9`, `atol = 1e-12`, NumPy as reference only, bit-identity with `seq` |

Measurements of Assignment 3 are **not** reused: `seq` and `par` are measured again in this session.

## Layout

```
.
├── README.md
├── pom.xml                       # Maven build, Hazelcast dependency, fat jar target/dmm.jar
├── requirements.txt
├── .gitignore
├── config/
│   ├── experiment.json           # sizes, workers, layouts, limits, JVM and Hazelcast settings
│   └── hazelcast.xml             # local cluster: discovery on 127.0.0.1, map and executor settings
├── src/main/java/                # Java, no packages
│   ├── Main.java                 # CLI + timing loop (one configuration per process): seq, par, dist
│   ├── Blocked.java              # sequential baseline (Assignment 2)
│   ├── ParallelBlocked.java      # shared-memory parallel version (Assignment 3)
│   ├── DistributedBlocked.java   # Hazelcast client: upload B, submit tasks, gather C
│   ├── StripTask.java            # distributed task: one strip of rows of C on a member
│   ├── WorkerNode.java           # one Hazelcast member (one JVM process)
│   ├── MatrixIO.java
│   └── Checksum.java
├── scripts/
│   ├── generate_inputs.py
│   ├── verify_correctness.py     # run BEFORE benchmarking
│   ├── run_benchmarks.py         # starts and stops the cluster for every dist configuration
│   └── make_report_assets.py     # tables and figures from the raw data
├── data/                         # manifest.json (the .bin inputs are not versioned)
└── results/                      # raw data, tables and figures
```

## Environment

| Component | Value |
|---|---|
| CPU | Intel Core Ultra 9 185H (hybrid: performance, efficient and low-power efficient cores) |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Java | OpenJDK 21.0.6 (Temurin-21.0.6+7), default JIT, G1 GC |
| Build | Apache Maven (3.10.0), maven-shade-plugin 3.6.0 |
| Hazelcast | 5.4.0 Community Edition (see note below) |
| Python | 3.11.9 (only for inputs, harness and analysis; never timed) |
| numpy / pandas / matplotlib / psutil | 2.2.6 / 3.0.0 / 3.10.6 / 7.0.0 |
| Power state | Plugged in, battery 100 %, default Windows power mode (same for the whole session) |

**Hazelcast version.** The course guide uses 5.4.2, but from 5.4 on patch releases (5.4.1, 5.4.2, ...)
are only published for the Enterprise Edition and are not on Maven Central. 5.4.0 is the closest
Community release; the API used here (members, `IMap`, `IExecutorService`, client) is the same.

## Distributed design

- **Cluster.** m `WorkerNode` processes on this machine form one cluster (`matmul`). Discovery is TCP/IP on
  127.0.0.1 only (ports 5701, 5702, ...; multicast and auto-detection disabled), see `config/hazelcast.xml`.
  Each member runs `StripTask`s on an executor with t threads (`WorkerNode` second argument).
  Workers p = m x t.
- **Client.** `Main --method dist` is a `HazelcastClient`, not a member: it stores no data and runs no
  tasks. Before measuring it checks that the cluster has exactly m members.
- **Partitioning (1-D by rows).** C is split into strips of 64 rows (`ceil(n/64)` tasks: 8, 16, 32 and 63
  for n = 500 ... 4000), the same strips as `par`. Strip s is submitted to member s mod m (round robin,
  balanced: at most one strip of difference between members).
  - **A** is scattered: each task carries its own 64 rows of A.
  - **B** is shared: the client writes it to the `IMap` `matrices` as strips of 64 rows. Each member
    reads all of B once per multiplication (bulk `getAll`) and caches it in its JVM for the other tasks
    of the same multiplication; every multiplication has a new job id, so a stale B is never reused.
  - **C** is gathered: each task returns its strip; the client copies the strips into C.
- **Same algorithm.** `StripTask.multiplyRows` uses the same tiles (bs = 64), i-k-j loop order and
  float64 as `Blocked` / `ParallelBlocked`; every result is bit-identical to `seq`.
- **No backups.** The `matrices` map has `backup-count 0`: the data is temporary and rebuilt every run,
  and backups would only add transfers between members.
- **Fault tolerance** was not studied: a member that dies makes the run fail (`member_died`).

## Requirements

Windows 11 (the harness reads the Windows peak working set; on Linux it falls back to RSS),
JDK 21 and Maven on the PATH, Python 3.11 with `requirements.txt`. Maven downloads Hazelcast on the
first build. Plug the laptop in, close other applications and do not use the machine during the benchmark.
Make sure no `WorkerNode` is running before a benchmark (`jps -l`).

## How to reproduce (PowerShell, repository root)

```powershell
# 0. Dependencies
python -m pip install -r requirements.txt

# 1. Build (fat jar with Hazelcast: target/dmm.jar)
mvn package

# 2. Inputs (about 330 MB) and check against Assignment 3
python scripts/generate_inputs.py --check-a3 path\to\assignment3\data\manifest.json

# 3. Correctness (before any measurement; must end with ALL TESTS PASSED)
python scripts/verify_correctness.py

# 4. Benchmarks (resumable; --dry-run lists the 42 configurations)
python scripts/run_benchmarks.py --dry-run
python scripts/run_benchmarks.py

# 5. Tables and figures
python scripts/make_report_assets.py
```

Run one distributed configuration by hand (one terminal per member, then the client):

```powershell
java -cp target/dmm.jar WorkerNode config/hazelcast.xml 1      # terminal 1
java -cp target/dmm.jar WorkerNode config/hazelcast.xml 1      # terminal 2
java -Xmx4g -cp target/dmm.jar Main --method dist --members 2 --a data\dense\A_n1000.bin --b data\dense\B_n1000.bin --n 1000 --reps 5 --warmup 3
```

`seq` and `par` run without a cluster:

```powershell
java -Xmx4g -cp target/dmm.jar Main --method par --threads 8 --a data\dense\A_n1000.bin --b data\dense\B_n1000.bin --n 1000
```

Output: `rep,batch,seconds_per_op,checksum,upload_s,execute_s,assemble_s,cleanup_s,b_fetches,max_b_s,
sum_compute_s,max_compute_s` (the last 8 columns only for `dist`), then `#key=value` metadata.
Quick test of the cluster without input files: `java -cp target/dmm.jar DistributedBlocked 300`.

## Experimental design in brief

| Aspect | Setting |
|---|---|
| Methods | `seq` (sequential blocked), `par` (threads, shared memory), `dist` (Hazelcast members) |
| Same algorithm | Same tiles (bs = 64), loop order, float64, row-major storage and input files |
| Sizes | n = 500, 1000, 2000, 4000 |
| Baselines | `seq`; `par` with p = 1, 2, 4, 8 threads |
| Strong scaling | `dist` with m = 1, 2, 4, 8 members x 1 thread (p = m, comparable with `par` p) |
| Layouts | 8 worker threads in total as 1x8, 2x4, 4x2, 8x1 (members x threads), n = 1000 and 4000 |
| Members per configuration | A fresh cluster is started before and stopped after every `dist` configuration |
| Inputs | Assignment 1-3 files, uniform [0, 1), seed `[42, n, tag]` |
| Timed region `seq` / `par` | Kernel only, as in Assignment 3 |
| Timed region `dist` | End-to-end on the client: upload of B + execute (send A, tasks, return C) + assembly of C. Phases reported separately; member-side B read and kernel times reported per task. Cluster start, client connection and cleanup of the map are outside and reported separately |
| Warm-up / batching | ≥ 3 repetitions and ≥ 300 ms; `seq`/`par` short kernels batched to ≈ 20 ms; `dist` one multiplication per sample |
| Repetitions | 5 (3 and no warm-up if the probe exceeds 30 s) |
| Summary statistic | Median; variability as IQR |
| Metrics | S(p) = T_seq / T, E(p) = S(p) / p, T_dist / T_par at the same p, phase shares (all from medians) |
| Data volume | Payload per multiplication computed from n and m (8n² each for B upload, A scatter, C return; m·8n² for B read by the members) |
| Memory | Peak working set of the client and of every member (psutil) + live heap after GC + array estimate 24n² bytes |
| Budgets | 300 s and 4 GB (client) per run, 2 GB heap per member; violations recorded as results |
| JVM arguments | `--add-opens`/`--add-exports` recommended by Hazelcast, for members and the `dist` client only |

Results, analysis and limitations are in the PDF report.

## Output files

`results/raw_measurements.csv`: one row per measured sample (`timestamp, experiment, method, members,
threads_per_member, total_threads, block, n, numeric_type, timer, rep, batch, seconds_per_op, checksum`,
the 8 `dist` phase columns, `warmup_iterations, reps_planned, oversubscribed, affinity, environment`).

`results/run_status.csv`: one row per configuration with status, probe time, peak memory of the client and
of the members (max and sum), live heap, tasks per member, cluster start and connection time, logical
processors, physical cores, limits, power state, Java and Hazelcast versions.

`results/correctness.csv`: the correctness tests (`seq`, `par` and `dist` on clusters 1x1, 2x1, 3x1, 2x2).

Generated by `make_report_assets.py`: `summary_time.csv`, `scaling.csv`, `phases.csv`, `layouts.csv`,
`data_volume.csv`, `memory.csv`, `setup.csv`, `limits.csv` and the figures `time_vs_workers.png`,
`speedup.png`, `efficiency.png`, `phases.png`, `layouts.png`, `memory.png`.

`logs/` (not versioned): Hazelcast output of every member.

## Use of generative AI

Generative AI (Claude) was used for planning, drafting and debugging code and documentation, in line
with the course policy. See the report for details.