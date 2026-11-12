#!/usr/bin/env python3
"""Tables and figures for Assignment 4 from results/raw_measurements.csv and results/run_status.csv.

Summary statistic: the MEDIAN of the measured samples of each configuration (seconds per
multiplication), with the interquartile range (IQR) as variability. Workers p = total worker threads
(par: threads; dist: members x threads per member).
  T_seq(n)          median time of seq (Blocked, 1 thread, kernel only)
  T_par(p, n)       median time of par with p threads (kernel only)
  T_dist(m x t, n)  median END-TO-END time of dist (upload of B + execute + assembly of C, client clock)
  S(p) = T_seq / T          speedup against the sequential baseline (also for dist)
  E(p) = S(p) / p           efficiency
  S_rel(m) = T_dist(1x1) / T_dist(m x 1)   scaling of dist against itself
  dist / par = T_dist(p) / T_par(p)        cost of distribution at the same number of worker threads

Data volume moved per multiplication (computed from n and m, 8 bytes per float64; the payload only,
without Hazelcast headers): upload of B 8n^2, scatter of A 8n^2, B read by the m members m*8n^2
(each member reads all of B once; the strips it owns itself are read locally), return of C 8n^2.

Usage: python scripts/make_report_assets.py [--in-dir results]
Writes into <in-dir>: summary_time.csv, scaling.csv, phases.csv, layouts.csv, data_volume.csv,
memory.csv, setup.csv, limits.csv and time_vs_workers.png, speedup.png, efficiency.png, phases.png,
layouts.png, memory.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OK = ["ok", "rep_over_time_limit"]
KEYS = ["method", "members", "threads_per_member", "total_threads", "n"]
PHASES = ["upload_s", "execute_s", "assemble_s", "cleanup_s", "max_b_s", "sum_compute_s", "max_compute_s"]
MB = 1e6


def summarise(df):
    g = df.groupby(KEYS)
    s = g["seconds_per_op"].agg(reps="count", median="median", q1=lambda x: x.quantile(.25),
                                q3=lambda x: x.quantile(.75), min="min", max="max").reset_index()
    s["iqr"] = s["q3"] - s["q1"]
    s["iqr_pct_of_median"] = 100 * s["iqr"] / s["median"]
    first = g[["experiment", "batch", "timer", "oversubscribed"]].first().reset_index(drop=True)
    s = pd.concat([s, first], axis=1)
    ph = g[PHASES].median().reset_index(drop=True).add_prefix("median_")
    return pd.concat([s, ph], axis=1)


def label(r):
    if r.method == "dist":
        return f"dist {r.members}x{r.threads_per_member}"
    return "seq" if r.method == "seq" else f"par {r.total_threads}"


def log2_axis(ax, ticks, xlabel, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    rd = Path(args.in_dir)

    df = pd.read_csv(rd / "raw_measurements.csv")
    st = pd.read_csv(rd / "run_status.csv")
    S = summarise(df)
    S.to_csv(rd / "summary_time.csv", index=False)
    sizes = sorted(S.n.unique())

    seq = S[S.method == "seq"].set_index("n")["median"]
    par = S[S.method == "par"]
    strong = S[(S.method == "dist") & (S.threads_per_member == 1)]
    dist1 = strong[strong.members == 1].set_index("n")["median"]

    # ------------------------------------------------------------------ scaling: par and dist (m x 1)
    rows = []
    for _, r in pd.concat([par, strong]).iterrows():
        p, n = r.total_threads, r.n
        sp = seq.get(n, float("nan")) / r["median"]
        same_par = par[(par.total_threads == p) & (par.n == n)]["median"]
        rows.append({"method": r.method, "workers": p, "n": n, "median_s": r["median"], "iqr_s": r.iqr,
                     "speedup_vs_seq": sp, "efficiency": sp / p,
                     "speedup_vs_dist_1x1": dist1.get(n, float("nan")) / r["median"] if r.method == "dist" else "",
                     "dist_over_par_same_workers": r["median"] / same_par.iloc[0]
                     if r.method == "dist" and len(same_par) else ""})
    sc = pd.DataFrame(rows).sort_values(["method", "n", "workers"])
    sc.to_csv(rd / "scaling.csv", index=False)

    # ------------------------------------------------------------------ phases of dist
    D = S[S.method == "dist"].copy()
    D["setup"] = D.apply(label, axis=1)
    for ph in ["upload_s", "execute_s", "assemble_s"]:
        D[ph.replace("_s", "_pct")] = 100 * D["median_" + ph] / D["median"]
    # average kernel time per worker thread (sum of the task kernel times / worker threads): the part of
    # "execute" that is computation if the load is balanced; the rest is sending A, reading B, returning C
    # and waiting. Only an indicator: the member-side times overlap with each other.
    D["est_member_compute_s"] = D["median_sum_compute_s"] / D["total_threads"]
    D["kernel_share_of_execute_pct"] = 100 * D["est_member_compute_s"] / D["median_execute_s"]
    D["tasks"] = (D.n + 63) // 64
    ph_cols = ["setup", "experiment", "members", "threads_per_member", "total_threads", "n", "median",
               "median_upload_s", "median_execute_s", "median_assemble_s", "median_cleanup_s", "upload_pct",
               "execute_pct", "assemble_pct", "median_max_b_s", "median_sum_compute_s", "median_max_compute_s",
               "est_member_compute_s", "kernel_share_of_execute_pct", "tasks"]
    D.sort_values(["threads_per_member", "n", "members"])[ph_cols].to_csv(rd / "phases.csv", index=False)

    # ------------------------------------------------------------------ layouts (constant total threads)
    lay = D[D.experiment.str.contains("layout")].copy()
    if len(lay):
        lay["par_same_threads_s"] = [par[(par.total_threads == p) & (par.n == n)]["median"].iloc[0]
                                     for p, n in zip(lay.total_threads, lay.n)]
        lay["dist_over_par"] = lay["median"] / lay["par_same_threads_s"]
        lay.sort_values(["n", "members"])[["setup", "members", "threads_per_member", "total_threads", "n", "median",
                                           "iqr", "par_same_threads_s", "dist_over_par", "median_upload_s",
                                           "median_execute_s", "upload_pct"]].to_csv(rd / "layouts.csv", index=False)

    # ------------------------------------------------------------------ data volume
    dv = D[["setup", "members", "threads_per_member", "n", "median_upload_s"]].copy()
    nn = 8 * dv.n.astype(float) ** 2
    dv["upload_B_MB"] = nn / MB
    dv["scatter_A_MB"] = nn / MB
    dv["B_read_by_members_MB"] = dv.members * nn / MB
    dv["return_C_MB"] = nn / MB
    dv["total_MB"] = dv[["upload_B_MB", "scatter_A_MB", "B_read_by_members_MB", "return_C_MB"]].sum(axis=1)
    dv["upload_throughput_MB_s"] = dv.upload_B_MB / dv.median_upload_s
    dv.sort_values(["threads_per_member", "n", "members"]).to_csv(rd / "data_volume.csv", index=False)

    # ------------------------------------------------------------------ memory, setup, limits
    st["setup"] = st.apply(label, axis=1)
    mem = st[st.status.isin(OK)][["setup", "method", "members", "threads_per_member", "total_threads", "n",
                                  "client_peak_memory_bytes", "member_peak_memory_max_bytes",
                                  "member_peak_memory_sum_bytes", "array_estimate_bytes"]].copy()
    mem["client_peak_mb"] = mem.client_peak_memory_bytes / MB
    mem["member_peak_max_mb"] = mem.member_peak_memory_max_bytes / MB
    mem["member_peak_sum_mb"] = mem.member_peak_memory_sum_bytes / MB
    mem["total_peak_mb"] = mem.client_peak_mb + mem.member_peak_sum_mb.fillna(0)
    mem["array_estimate_mb"] = mem.array_estimate_bytes / MB
    mem.drop(columns=[c for c in mem.columns if c.endswith("_bytes")]).sort_values(
        ["n", "method", "total_threads", "members"]).to_csv(rd / "memory.csv", index=False)

    st[st.method == "dist"][["setup", "n", "cluster_start_s", "connect_s", "tasks_per_member"]].to_csv(
        rd / "setup.csv", index=False)

    lim = []
    for k, g in st.groupby(["method", "members", "threads_per_member", "total_threads"]):
        ok = g[g.status.isin(OK)].n
        bad = g[~g.status.isin(OK)]
        lim.append({"setup": label(g.iloc[0]), "largest_tested_ok_n": ok.max() if len(ok) else "",
                    "largest_attempted_n": g.n.max(),
                    "first_failure": "" if bad.empty else f"{bad.iloc[0].status} at n={bad.iloc[0].n}"})
    pd.DataFrame(lim).to_csv(rd / "limits.csv", index=False)

    # ------------------------------------------------------------------ figures
    workers = sorted(sc.workers.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8))
    for ax, n in zip(axes, sizes):
        for meth, mk in [("par", "o-"), ("dist", "s-")]:
            d = sc[(sc.method == meth) & (sc.n == n)]
            ax.errorbar(d.workers, d.median_s, yerr=d.iqr_s / 2, fmt=mk, capsize=3,
                        label="par (threads)" if meth == "par" else "dist (members x 1)")
        ax.axhline(seq[n], color="k", ls="--", lw=1, label="seq")
        ax.set_yscale("log")
        log2_axis(ax, workers, "workers p", "median time (s)", f"n = {n}")
    fig.tight_layout(); fig.savefig(rd / "time_vs_workers.png", dpi=150); plt.close(fig)

    for col, fname, ylab, ideal in [("speedup_vs_seq", "speedup.png", "speedup vs seq", True),
                                    ("efficiency", "efficiency.png", "efficiency", False)]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, meth in zip(axes, ["par", "dist"]):
            for n in sizes:
                d = sc[(sc.method == meth) & (sc.n == n)]
                ax.plot(d.workers, d[col], "o-", label=f"n={n}")
            if ideal:
                ax.plot(workers, workers, "k:", lw=1, label="ideal")
            else:
                ax.axhline(1, color="k", ls=":", lw=1)
            log2_axis(ax, workers, "workers p", ylab,
                      "par (threads, shared memory)" if meth == "par" else "dist (Hazelcast members x 1 thread)")
        fig.tight_layout(); fig.savefig(rd / fname, dpi=150); plt.close(fig)

    sd = D[D.threads_per_member == 1].sort_values(["n", "members"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.8))
    for ax, n in zip(axes, sizes):
        d = sd[sd.n == n]
        x = [str(m) for m in d.members]
        b = 0
        for ph, c in [("median_upload_s", "tab:orange"), ("median_execute_s", "tab:blue"),
                      ("median_assemble_s", "tab:green")]:
            ax.bar(x, d[ph], bottom=b, color=c, label=ph.replace("median_", "").replace("_s", ""))
            b = b + d[ph].values
        ax.plot(x, d.est_member_compute_s, "k_", ms=22, mew=2, label="kernel per member (est.)")
        ax.set_xlabel("members (1 thread each)"); ax.set_ylabel("median time (s)"); ax.set_title(f"n = {n}")
        ax.grid(True, axis="y", alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(rd / "phases.png", dpi=150); plt.close(fig)

    if len(lay):
        lsz = sorted(lay.n.unique())
        fig, axes = plt.subplots(1, len(lsz), figsize=(5 * len(lsz), 3.8))
        axes = axes if len(lsz) > 1 else [axes]
        for ax, n in zip(axes, lsz):
            d = lay[lay.n == n].sort_values("members")
            x = [f"{m}x{t}" for m, t in zip(d.members, d.threads_per_member)]
            ax.bar(x, d.median_upload_s, color="tab:orange", label="upload")
            ax.bar(x, d.median_execute_s, bottom=d.median_upload_s, color="tab:blue", label="execute")
            ax.errorbar(x, d["median"], yerr=d.iqr / 2, fmt="none", ecolor="k", capsize=3)
            ax.axhline(d.par_same_threads_s.iloc[0], color="tab:red", ls="--",
                       label=f"par {d.total_threads.iloc[0]} threads")
            ax.set_xlabel("members x threads per member"); ax.set_ylabel("median time (s)")
            ax.set_title(f"n = {n}, {d.total_threads.iloc[0]} worker threads in total")
            ax.grid(True, axis="y", alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(rd / "layouts.png", dpi=150); plt.close(fig)

    nmax = max(sizes)
    m4 = mem[(mem.n == nmax) & (mem.method != "seq") & ((mem.method == "par") | (mem.threads_per_member == 1))]
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, mk in [("par", "o-"), ("dist", "s-")]:
        d = m4[m4.method == meth].sort_values("total_threads")
        ax.plot(d.total_threads, d.total_peak_mb / 1000, mk, label=f"{meth}: all processes")
    d = m4[m4.method == "dist"].sort_values("total_threads")
    ax.plot(d.total_threads, d.client_peak_mb / 1000, "s--", alpha=.6, label="dist: client only")
    log2_axis(ax, workers, "workers p", "peak memory (GB)", f"Peak memory, n = {nmax}")
    fig.tight_layout(); fig.savefig(rd / "memory.png", dpi=150); plt.close(fig)

    print(f"Wrote tables and figures into {rd}")


if __name__ == "__main__":
    main()