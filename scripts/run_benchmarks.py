#!/usr/bin/env python3
"""Benchmark harness for Assignment 4 (seq and par baselines vs distributed Hazelcast multiplication).

Same protocol as Assignments 1-3:
  * every configuration runs in its own client JVM process (Main);
  * probe run (1 repetition, no warm-up) -> time of one multiplication and peak memory;
  * timing run: warm-up (>= N repetitions AND >= warmup_min_ms, excluded) + measured repetitions,
    all inside one process. seq / par kernels are batched automatically (time reported PER operation);
    dist always uses one multiplication per sample;
  * if the probe takes > costly_s seconds: 3 repetitions and no warm-up (reported);
  * limits: time_s per multiplication and mem_bytes of peak memory of the client process. A violation is
    recorded as a result (timeout / memory_exceeded) without any invented time; larger n of the same
    series is skipped;
  * checksums of all configurations on the same input are compared and mismatches are reported.

New in Assignment 4 (method dist):
  * for every dist configuration a FRESH cluster of `members` WorkerNode processes is started, each with
    `threads_per_member` executor threads, before the probe; it is stopped after the timing run.
    Members are started one after the other and each must report the expected cluster size;
  * memory: peak working set of the client (limit applies) and of every member (max and sum reported).
    On Windows the peak working set is the peak since the process started, so for members it covers
    the probe and the timing run of the configuration;
  * the member logs (Hazelcast output) are written to logs/.
seq and par are run exactly as in Assignment 3 (no Hazelcast JVM arguments).

Configurations come from config/experiment.json: for every n of "sizes", seq, par with every
"baselines.par_threads", dist with every "strong_scaling.members" (threads_per_member fixed), and for
the n of "layouts.sizes", dist with every (members, threads_per_member) of "layouts.members_x_threads".
Duplicates (e.g. 8 x 1 in both lists) are run once and tagged with both experiments.

Outputs (append mode, resumable; default results/):
  raw_measurements.csv   one row per measured sample
  run_status.csv         one row per configuration

Usage (repo root, after mvn package):
  python scripts/run_benchmarks.py --dry-run
  python scripts/run_benchmarks.py --out-dir results/test --sizes 500 --members 1 2
  python scripts/run_benchmarks.py
"""
import argparse
import atexit
import csv
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA, JAR, CONFIG = ROOT / "data" / "dense", ROOT / "target" / "dmm.jar", ROOT / "config" / "experiment.json"
HZ_XML, LOGS = ROOT / "config" / "hazelcast.xml", ROOT / "logs"
LOGICAL = psutil.cpu_count(logical=True)
PHYSICAL = psutil.cpu_count(logical=False)

PHASES = ["upload_s", "execute_s", "assemble_s", "cleanup_s", "b_fetches", "max_b_s", "sum_compute_s", "max_compute_s"]
RAW_HEADER = ["timestamp", "experiment", "method", "members", "threads_per_member", "total_threads", "block", "n",
              "numeric_type", "timer", "rep", "batch", "seconds_per_op", "checksum"] + PHASES + \
             ["warmup_iterations", "reps_planned", "oversubscribed", "affinity", "environment"]
STATUS_HEADER = ["timestamp", "experiment", "method", "members", "threads_per_member", "total_threads", "block", "n",
                 "status", "probe_seconds", "client_peak_memory_bytes", "member_peak_memory_max_bytes",
                 "member_peak_memory_sum_bytes", "memory_metric", "live_heap_bytes_after_gc", "array_estimate_bytes",
                 "reps", "warmup_planned", "warmup_iterations", "tasks_per_multiplication", "tasks_per_member",
                 "cluster_start_s", "connect_s", "active_java_threads", "available_processors", "logical_processors",
                 "physical_cores", "oversubscribed", "time_limit_s", "mem_limit_bytes", "member_xmx", "affinity",
                 "power_plugged", "battery_percent", "java_version", "hazelcast_version", "environment", "note"]


# ----------------------------------------------------------------------------- configurations
def build_configs(cfg, sizes=None, methods=None, members=None, experiments=None):
    out = {}

    def add(exp, method, n, m, tpm, threads):
        c = {"method": method, "n": n, "members": m, "tpm": tpm, "threads": threads, "experiment": exp}
        k = key_of(c)
        if k in out:
            if exp not in out[k]["experiment"].split("+"):
                out[k]["experiment"] += "+" + exp
        else:
            out[k] = c

    base, strong, lay = cfg["baselines"], cfg["strong_scaling"], cfg.get("layouts", {})
    for n in cfg["sizes"]:
        if base.get("seq", True):
            add("baseline", "seq", n, 0, 0, 1)
        for p in base.get("par_threads", []):
            add("baseline", "par", n, 0, 0, p)
        for m in strong["members"]:
            t = strong["threads_per_member"]
            add("strong_scaling", "dist", n, m, t, m * t)
        if n in lay.get("sizes", []):
            for m, t in lay["members_x_threads"]:
                add("layout", "dist", n, m, t, m * t)

    res = list(out.values())
    if sizes:
        res = [c for c in res if c["n"] in sizes]
    if methods:
        res = [c for c in res if c["method"] in methods]
    if members:
        res = [c for c in res if c["method"] != "dist" or c["members"] in members]
    if experiments:
        res = [c for c in res if set(c["experiment"].split("+")) & set(experiments)]
    return res


def key_of(c):
    return f"{c['method']}|m{c['members']}|t{c['tpm']}|p{c['threads']}|n{c['n']}"


def series(c):
    return (c["method"], c["members"], c["tpm"], c["threads"])


# ----------------------------------------------------------------------------- memory and processes
def peak_of(ps):
    """Peak working set (Windows) or current RSS (elsewhere) of a process and its children."""
    total = 0
    try:
        procs = [ps] + ps.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            mi = p.memory_info()
            total += getattr(mi, "peak_wset", mi.rss)
        except psutil.Error:
            pass
    return total


def kill_tree(proc):
    try:
        for ch in psutil.Process(proc.pid).children(recursive=True):
            ch.kill()
    except psutil.Error:
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_monitored(cmd, timeout_s, mem_limit, members=None):
    """Runs the client; polls its memory (limit applies) and the memory of the cluster members.
    Returns status, stdout, stderr, client peak, list of member peaks."""
    members = members or []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    ps = psutil.Process(proc.pid)
    mps = [psutil.Process(m.pid) for m in members]
    mpeaks = [0] * len(members)
    start, peak, status = time.perf_counter(), 0, "ok"
    # stdout/stderr are read in threads so that a verbose client can never block on a full pipe
    out_buf, err_buf = [], []
    t_out = threading.Thread(target=lambda: out_buf.append(proc.stdout.read()), daemon=True)
    t_err = threading.Thread(target=lambda: err_buf.append(proc.stderr.read()), daemon=True)
    t_out.start(); t_err.start()
    while proc.poll() is None:
        peak = max(peak, peak_of(ps))
        for i, mp in enumerate(mps):
            mpeaks[i] = max(mpeaks[i], peak_of(mp))
        if peak > mem_limit:
            status = "memory_exceeded"
            break
        if time.perf_counter() - start > timeout_s:
            status = "timeout"
            break
        if any(m.poll() is not None for m in members):
            status = "member_died"
            break
        time.sleep(0.01)
    for i, mp in enumerate(mps):                         # last sample after the client ended
        mpeaks[i] = max(mpeaks[i], peak_of(mp))
    if status != "ok":
        kill_tree(proc)
    proc.wait()
    t_out.join(); t_err.join()
    out, err = "".join(out_buf), "".join(err_buf)
    if status == "ok" and proc.returncode != 0:
        status = "error"
    return status, out, err, peak, mpeaks


class Cluster:
    """p WorkerNode processes on this machine, started one by one and checked to form one cluster."""
    alive = []                                           # for the atexit safety net

    def __init__(self, members, tpm, hz, tag):
        self.members, self.tpm, self.hz, self.tag = members, tpm, hz, tag
        self.procs, self.logs = [], []

    def start(self):
        LOGS.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        for i in range(1, self.members + 1):
            log = open(LOGS / f"{self.tag}_member{i}.log", "w")
            cmd = ["java", f"-Xmx{self.hz['member_xmx']}", *self.hz["jvm_args"], "-cp", str(JAR),
                   "WorkerNode", str(HZ_XML), str(self.tpm)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT)
            self.procs.append(p); self.logs.append(log); Cluster.alive.append(p)
            size = self._wait_size(p, self.hz.get("member_start_timeout_s", 60))
            if size != i:
                self.stop()
                sys.exit(f"member {i} sees a cluster of {size} members (expected {i}). Is another "
                         f"WorkerNode still running? Check with: jps -l")
        return time.perf_counter() - t0

    @staticmethod
    def _wait_size(p, timeout):
        q = queue.Queue()

        def reader():
            for line in p.stdout:
                q.put(line)
        threading.Thread(target=reader, daemon=True).start()
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if p.poll() is not None:
                return -1
            try:
                line = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if line.startswith("#cluster_size="):
                return int(line.strip().split("=")[1])
        return -1

    def stop(self):
        for p in self.procs:
            kill_tree(p)
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            if p in Cluster.alive:
                Cluster.alive.remove(p)
        for f in self.logs:
            f.close()
        time.sleep(1)                                    # let the OS release the ports


@atexit.register
def _kill_leftovers():
    for p in list(Cluster.alive):
        kill_tree(p)


# ----------------------------------------------------------------------------- output helpers
def parse(out):
    rows, meta = [], {}
    for line in out.strip().splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition("=")
            meta[k] = v
        elif line and line[0].isdigit():
            p = line.split(",")
            r = {"rep": int(p[0]), "batch": int(p[1]), "seconds": float(p[2]), "checksum": float(p[3])}
            for name, v in zip(PHASES, p[4:12]):
                r[name] = v
            rows.append(r)
    return rows, meta


def append(path, header, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def load_status(path):
    done, blocked = set(), set()
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                c = {"method": r["method"], "members": int(r["members"]), "tpm": int(r["threads_per_member"]),
                     "threads": int(r["total_threads"]), "n": int(r["n"])}
                done.add(key_of(c))
                if r["status"] != "ok":
                    blocked.add(series(c))
    return done, blocked


def power_state():
    try:
        b = psutil.sensors_battery()
    except Exception:
        b = None
    if b is None:
        return "unknown", ""
    return str(b.power_plugged), f"{b.percent:.0f}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--sizes", type=int, nargs="+", help="restrict to these n")
    ap.add_argument("--methods", nargs="+", help="restrict to these methods (seq par dist)")
    ap.add_argument("--members", type=int, nargs="+", help="restrict dist to these member counts")
    ap.add_argument("--experiments", nargs="+", help="restrict to baseline / strong_scaling / layout")
    ap.add_argument("--reps", type=int, help="override measured repetitions")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true", help="list the configurations and exit")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    lim, jv, hz, block = cfg["limits"], cfg["java"], cfg["hazelcast"], cfg["block"]
    aff, env = cfg.get("affinity", "none"), cfg.get("environment", "")
    reps_cfg = args.reps or jv["reps"]
    time_lim, mem_lim, costly = lim["time_s"], lim["mem_bytes"], lim["costly_s"]

    configs = build_configs(cfg, args.sizes, args.methods, args.members, args.experiments)
    print(f"Logical processors: {LOGICAL}, physical cores: {PHYSICAL}, affinity: {aff}")
    if args.dry_run:
        for c in configs:
            note = " (oversubscribed)" if c["threads"] > LOGICAL else ""
            print(f"{key_of(c):32s} {c['experiment']}{note}")
        print(f"{len(configs)} configurations")
        return
    if not JAR.exists():
        sys.exit("missing target/dmm.jar: run 'mvn package' first (see README)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, status_path = out_dir / "raw_measurements.csv", out_dir / "run_status.csv"
    done, blocked = load_status(status_path)
    checks = {}

    for c in configs:
        k, n, method = key_of(c), c["n"], c["method"]
        m, tpm, p = c["members"], c["tpm"], c["threads"]
        over = p > LOGICAL
        ts = datetime.now().isoformat(timespec="seconds")
        if k in done:
            print(f"[skip] {k} already done")
            continue
        plugged, batt = power_state()
        cluster_s = ""

        def status_row(status, probe_s="", peak="", mpeaks=None, meta=None, reps="", wu="", wu_it="", note=""):
            meta, mpeaks = meta or {}, mpeaks or []
            return [ts, c["experiment"], method, m, tpm, p, block, n, status, probe_s, peak,
                    max(mpeaks) if mpeaks else "", sum(mpeaks) if mpeaks else "", "peak_working_set",
                    meta.get("live_heap_bytes_after_gc", ""), meta.get("array_estimate_bytes", ""), reps, wu, wu_it,
                    meta.get("tasks_per_multiplication", ""), meta.get("tasks_per_member", ""),
                    f"{cluster_s:.3f}" if cluster_s != "" else "", meta.get("connect_s", ""),
                    meta.get("active_java_threads", ""), meta.get("available_processors", ""), LOGICAL, PHYSICAL,
                    over, time_lim, mem_lim, hz["member_xmx"] if method == "dist" else "", aff, plugged, batt,
                    meta.get("java_version", ""), hz["version"] if method == "dist" else "", env, note]

        if series(c) in blocked:
            append(status_path, STATUS_HEADER, status_row("skipped_after_failure"))
            print(f"[skip] {k}: a smaller size of this series already failed")
            continue
        a, b = DATA / f"A_n{n}.bin", DATA / f"B_n{n}.bin"
        if not (a.exists() and b.exists()):
            sys.exit(f"missing input {a.name}/{b.name}: run scripts/generate_inputs.py")

        jvm = ["java", f"-Xmx{jv['xmx']}"] + (hz["jvm_args"] if method == "dist" else [])
        base = jvm + ["-cp", str(JAR), "Main", "--method", method, "--a", str(a), "--b", str(b),
                      "--n", str(n), "--block", str(block)]
        if method == "par":
            base += ["--threads", str(p)]
        if method == "dist":
            base += ["--members", str(m), "--cluster", hz["cluster"], "--address", hz["address"]]

        cluster = None
        try:
            if method == "dist":
                print(f"[cluster] {m} member(s) x {tpm} thread(s) ...", flush=True)
                cluster = Cluster(m, tpm, hz, f"n{n}_m{m}_t{tpm}")
                cluster_s = cluster.start()
            members = cluster.procs if cluster else []

            print(f"[probe] {k} ...", flush=True)
            st, out, err, peak, mpeaks = run_monitored(
                base + ["--reps", "1", "--warmup", "0", "--warmup-min-ms", "0", "--batch", "1"],
                time_lim + 120, mem_lim, members)
            if st == "error" and method != "dist":
                sys.exit(f"{k} failed: {err.strip()[-800:]}")
            rows, meta = parse(out) if st == "ok" else ([], {})
            probe_s = rows[0]["seconds"] if rows else ""
            if st == "ok" and probe_s > time_lim:
                st = "timeout"
            if st != "ok":
                append(status_path, STATUS_HEADER, status_row(st, probe_s, peak, mpeaks, meta, 0, 0,
                                                              note=err.strip()[-300:].replace("\n", " ")))
                blocked.add(series(c))
                print(f"  -> {st} (client peak {peak / 1e6:.0f} MB) {err.strip()[-300:] if st == 'error' else ''}")
                continue

            if probe_s > costly:
                reps, warmup, wmin = 3, 0, 0
            else:
                reps, warmup, wmin = reps_cfg, jv["warmup"], jv["warmup_min_ms"]
            print(f"  probe {probe_s:.4f}s, client peak {peak / 1e6:.0f} MB -> {reps} reps + >= {warmup} warm-up",
                  flush=True)
            budget = time_lim * (reps + warmup + 1) + 120 + wmin / 1000
            st2, out, err, peak2, mpeaks2 = run_monitored(
                base + ["--reps", str(reps), "--warmup", str(warmup), "--warmup-min-ms", str(wmin),
                        "--target-ms", str(jv["target_ms"])], budget, mem_lim, members)
            peak = max(peak, peak2)
            mpeaks = [max(x, y) for x, y in zip(mpeaks, mpeaks2)] if mpeaks else mpeaks2
            if st2 != "ok":
                append(status_path, STATUS_HEADER, status_row(st2 + "_timing_run", probe_s, peak, mpeaks, meta, reps,
                                                              warmup, note=err.strip()[-300:].replace("\n", " ")))
                blocked.add(series(c))
                print(f"  -> {st2} in timing run {err.strip()[-300:] if st2 == 'error' else ''}")
                continue
        finally:
            if cluster:
                cluster.stop()

        rows, meta2 = parse(out)
        timer = "end_to_end_client" if method == "dist" else "kernel_only"
        for r in rows:
            append(raw_path, RAW_HEADER, [ts, c["experiment"], method, m, tpm, p, block, n, "float64", timer,
                   r["rep"], r["batch"], f"{r['seconds']:.9e}", f"{r['checksum']:.12e}"] +
                   [r[x] for x in PHASES] + [meta2.get("warmup_iterations", ""), reps, over, aff, env])
        if len({r["checksum"] for r in rows}) != 1:
            print(f"  WARNING: checksum changes between repetitions for {k}")
        checks.setdefault(n, {})[k] = rows[0]["checksum"]
        if len(set(checks[n].values())) != 1:
            print(f"  WARNING: checksums differ across configurations for n={n}: {checks[n]}")
        final = "ok" if max(r["seconds"] for r in rows) <= time_lim else "rep_over_time_limit"
        append(status_path, STATUS_HEADER, status_row(final, probe_s, peak, mpeaks, meta2 or meta, reps, warmup,
                                                      meta2.get("warmup_iterations", "")))
        t = sorted(r["seconds"] for r in rows)
        extra = f", members peak max {max(mpeaks) / 1e6:.0f} MB" if mpeaks else ""
        print(f"  median {t[len(t) // 2]:.6g} s (batch {rows[0]['batch']}){extra}")


if __name__ == "__main__":
    main()