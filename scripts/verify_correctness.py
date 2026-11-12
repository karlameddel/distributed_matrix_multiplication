#!/usr/bin/env python3
"""Correctness validation for Assignment 4 (run BEFORE benchmarking).

Every method is checked against NumPy (A @ B), which is used ONLY as a reference:
  seq   Blocked (sequential baseline of Assignments 2 and 3)
  par   ParallelBlocked with several thread counts, including 1 and oversubscription
  dist  DistributedBlocked on real local Hazelcast clusters (members x threads per member):
        1x1, 2x1, 3x1 (strips not divisible by the member count) and 2x2 (several threads per member).
        With n = 1 and 10 there is a single strip, so some members receive no task.
Cases: a hand-computed 2x2 product; A*I = A, I*B = B, A*0 = 0; random dense matrices (same
generator and seeds as the benchmark inputs); n = 1 and sizes that the block size does not divide.
Block sizes 7, 64 and 300 (does not divide n / default / larger than n).

Pass criterion (same as Assignments 1-3): |x - ref| <= ATOL + RTOL*|ref| for every entry,
RTOL = 1e-9, ATOL = 1e-12. All entries are >= 0, so there is no cancellation and the rounding error
of a sum of n terms (about n*2^-53 relative) is far below the tolerance.

Safe access to shared data and correct distribution: every par and dist result must also be
BIT-IDENTICAL to the seq result with the same block size (each C[i][j] receives the same additions in
the same order). par and dist run 3 repetitions in the same process whose checksums must be identical;
for dist every repetition is a new job, so this also checks that a member never reuses the B of a
previous multiplication.

Usage: python scripts/verify_correctness.py [--sizes 1 10 100 200] [--threads 1 2 3 8 22 44]
                                            [--clusters 1x1 2x1 3x1 2x2]
Writes results/correctness.csv; exit code 1 on any failure.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_inputs import dense_matrix  # noqa: E402
from run_benchmarks import JAR, CONFIG, Cluster  # noqa: E402

OUT_CSV = ROOT / "results" / "correctness.csv"
RTOL, ATOL = 1e-9, 1e-12
BLOCKS = [7, 64, 300]


def run(method, threads, members, a, b, n, out, block, hz):
    reps = "1" if method == "seq" else "3"
    cmd = ["java"] + (hz["jvm_args"] if method == "dist" else []) + [
        "-cp", str(JAR), "Main", "--method", method, "--a", str(a), "--b", str(b), "--n", str(n),
        "--block", str(block), "--reps", reps, "--warmup", "0", "--warmup-min-ms", "0", "--batch", "1",
        "--out", str(out)]
    if method == "par":
        cmd += ["--threads", str(threads)]
    if method == "dist":
        cmd += ["--members", str(members), "--cluster", hz["cluster"], "--address", hz["address"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-300:])
    sums = [float(l.split(",")[3]) for l in r.stdout.splitlines() if l and l[0].isdigit()]
    c = np.fromfile(out, dtype="<f8")
    if c.size != n * n:
        raise RuntimeError(f"output has {c.size} values, expected {n * n}")
    return c.reshape(n, n), sums


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 200])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 8, 22, 44])
    ap.add_argument("--clusters", nargs="+", default=["1x1", "2x1", "3x1", "2x2"],
                    help="members x threads per member")
    args = ap.parse_args()
    if not JAR.exists():
        sys.exit("missing target/dmm.jar: run 'mvn package' first (see README)")
    hz = json.loads(CONFIG.read_text())["hazelcast"]
    clusters = [tuple(int(x) for x in s.split("x")) for s in args.clusters]

    cases = [("known_2x2", 2, np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]]),
              np.array([[19., 22.], [43., 50.]]))]                       # reference computed by hand
    for n in args.sizes:
        A, B = dense_matrix(n, 0), dense_matrix(n, 1)
        I, Z = np.eye(n), np.zeros((n, n))
        cases += [("A_times_identity", n, A, I, A), ("identity_times_B", n, I, B, B),
                  ("A_times_zero", n, A, Z, Z), ("random_dense", n, A, B, A @ B)]

    rows, all_ok, seq_c = [], True, {}

    def check(tmp, method, p, m, tpm, label):
        nonlocal all_ok
        a_p, b_p, c_p = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
        for ci, (case, n, A, B, ref) in enumerate(cases):
            A.astype("<f8").tofile(a_p)
            B.astype("<f8").tofile(b_p)
            for bs in BLOCKS:
                try:
                    c, sums = run(method, p, m, a_p, b_p, n, c_p, bs, hz)
                    err = np.abs(c - ref)
                    nz = ref != 0
                    abs_e = float(err.max())
                    rel = float((err[nz] / np.abs(ref[nz])).max()) if nz.any() else 0.0
                    ok = bool(np.all(err <= ATOL + RTOL * np.abs(ref)))
                    if method == "seq":
                        seq_c[(ci, bs)] = c
                        same = ""
                    else:
                        same = bool(np.array_equal(c, seq_c[(ci, bs)]))
                        ok &= same and len(set(sums)) == 1
                except Exception as e:
                    abs_e = rel = float("nan"); ok = False; same = ""
                    print(f"  ERROR {label}/{case}/n={n}/bs={bs}: {e}")
                all_ok &= ok
                rows.append([case, n, method, m, tpm, p, bs, f"{abs_e:.3e}", f"{rel:.3e}", same,
                             RTOL, ATOL, "PASS" if ok else "FAIL"])
                print(f"[{'PASS' if ok else 'FAIL'}] {case:17s} n={n:<4d} bs={bs:<4d} {label:14s}"
                      f" max_abs={abs_e:.2e} max_rel={rel:.2e} bit_identical_to_seq={same if same != '' else '-'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check(tmp, "seq", 1, 0, 0, "seq")                                 # first: the bit-identity reference
        for t in args.threads:
            check(tmp, "par", t, 0, 0, f"par p={t}")
        for m, tpm in clusters:
            print(f"[cluster] {m} member(s) x {tpm} thread(s) ...", flush=True)
            cl = Cluster(m, tpm, hz, f"correctness_m{m}_t{tpm}")
            try:
                cl.start()
                check(tmp, "dist", m * tpm, m, tpm, f"dist {m}x{tpm}")
            finally:
                cl.stop()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "method", "members", "threads_per_member", "total_threads", "block",
                    "max_abs_error", "max_rel_error", "bit_identical_to_seq", "rtol", "atol", "result"])
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ({len(rows)} tests)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()