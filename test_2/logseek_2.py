"""
LogSeek: a learned time index for server logs.

Machine learning
  * A small neural network (one hidden layer, tanh) is trained with Adam to map
    a timestamp to the position of its line in the file.
  * Each time bucket stores the largest prediction error over its own lines.
    A query searches only [prediction - error, prediction + error] inside its
    bucket, so a better-trained model gives a smaller search window.
  * Every answer is verified. If the window misses, the search falls back to
    the whole file, so results are always exact.
  * Appended lines fine-tune the network (warm-started Adam) and refresh the
    bucket error table.

Usage
  python logseek.py generate  server.log [--append-out new_lines.log] [--force]
  python logseek.py build     server.log
  python logseek.py demo      server.log
  python logseek.py query     server.log --date 2026-11-10 --start 14:02:10 --end 14:02:45 [--all]
  python logseek.py append    server.log new_lines.log
  python logseek.py benchmark server.log

Requirements: pip install numpy matplotlib
"""
import argparse
import os
import time

import numpy as np

SEED = 42
HIDDEN = 16              # hidden units in the timestamp -> position network
BUCKETS = 4096           # time buckets, each with its own error bound
HEADROOM = 1.25          # capacity above current data, so appends keep the same mapping
PARSE_CHUNK = 64 << 20   # bytes parsed per pass, so large logs fit in memory
ERROR_TAG = np.frombuffer(b"ERROR", dtype=np.uint8)


# =====================================================================
# 1. PARSING
# =====================================================================

def parse_log_bytes(buf, base_offset=0):
    """Returns (timestamps in ms, byte offset of each line, ERROR flag per line)."""
    nl = np.flatnonzero(buf == 10)
    starts = np.concatenate(([0], nl + 1)).astype(np.int64)
    starts = starts[starts < buf.size]
    last = buf.size - 1

    raw = buf[np.minimum(starts[:, None] + np.arange(23), last)].copy()
    if not np.all(raw[:, 10] == ord(" ")):
        raise ValueError("lines must start with 'YYYY-MM-DD HH:MM:SS.mmm'")
    raw[:, 10] = ord("T")
    ts = raw.view("S23").ravel().astype("U23").astype("datetime64[ms]").astype(np.int64)

    level = buf[np.minimum(starts[:, None] + 24 + np.arange(5), last)]
    is_err = np.all(level == ERROR_TAG, axis=1)
    return ts, starts + base_offset, is_err


def parse_log_file(path, chunk=PARSE_CHUNK):
    """
    Parses a log in chunks, so large files don't need one huge index array.
    Returns (timestamps, line byte offsets, ERROR flags, file size).
    """
    ts_parts, off_parts, err_parts = [], [], []
    pos = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            if not data.endswith(b"\n"):
                data += f.readline()          # finish the partial line at the chunk boundary
            buf = np.frombuffer(data, dtype=np.uint8)
            ts, offs, err = parse_log_bytes(buf, base_offset=pos)
            ts_parts.append(ts)
            off_parts.append(offs)
            err_parts.append(err)
            pos += len(data)
    if not ts_parts:
        raise ValueError(f"'{path}' is empty")
    ts = np.concatenate(ts_parts)
    if np.any(np.diff(ts) < 0):
        raise ValueError("log must be in time order")
    return ts, np.concatenate(off_parts), np.concatenate(err_parts), pos


# =====================================================================
# 2. THE LEARNED MODEL
# =====================================================================

class TimeNet:
    """Maps u (normalized time, about 0..1) to y (normalized line position, 0..1)."""

    def __init__(self, hidden=HIDDEN, seed=SEED):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0.0, 4.0, hidden)
        self.P = {
            "W1": W1,
            "b1": -W1 * np.linspace(0.0, 1.0, hidden),   # tanh kinks spread over [0, 1]
            "W2": rng.normal(0.0, 0.3, hidden),
            "b2": np.zeros(1),
        }

    def _forward(self, u):
        P = self.P
        h = np.tanh(u[:, None] * P["W1"] + P["b1"])
        return h, h @ P["W2"] + P["b2"][0]

    def predict(self, u, chunk=200_000):
        out = np.empty(u.size)
        for i in range(0, u.size, chunk):
            out[i:i + chunk] = self._forward(u[i:i + chunk])[1]
        return out

    def fit(self, u, y, steps=1500, lr=0.01, batch=20_000, seed=SEED):
        """Gradient descent with Adam on mean squared error, using random mini-batches."""
        rng = np.random.default_rng(seed)
        P = self.P
        m = {k: np.zeros_like(v) for k, v in P.items()}
        s = {k: np.zeros_like(v) for k, v in P.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, steps + 1):
            idx = rng.integers(0, u.size, min(batch, u.size))
            uu, yy = u[idx], y[idx]
            h, pred = self._forward(uu)

            dp = 2.0 * (pred - yy) / uu.size                 # dLoss/dpred
            dz = dp[:, None] * P["W2"] * (1.0 - h * h)       # backprop through tanh
            grad = {
                "W1": (dz * uu[:, None]).sum(axis=0),
                "b1": dz.sum(axis=0),
                "W2": h.T @ dp,
                "b2": np.array([dp.sum()]),
            }
            for k in P:
                m[k] = beta1 * m[k] + (1 - beta1) * grad[k]
                s[k] = beta2 * s[k] + (1 - beta2) * grad[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                s_hat = s[k] / (1 - beta2 ** step)
                P[k] -= lr * m_hat / (np.sqrt(s_hat) + eps)


# =====================================================================
# 3. THE INDEX
# =====================================================================

def _bsearch(arr, t, lo, hi):
    """First index in [lo, hi] with arr[index] >= t, or hi if none."""
    while lo < hi:
        m = (lo + hi) >> 1
        if arr[m] < t:
            lo = m + 1
        else:
            hi = m
    return lo


class LogSeekIndex:
    def __init__(self, ts, offsets, is_err, size, state=None, steps=1500):
        self.ts = ts
        self.offsets = offsets
        self.is_err = is_err
        self.size = int(size)
        self.net = TimeNet()
        if state is None:
            self._set_capacity()
            self._train(steps, lr=0.01)
        else:
            self.t0 = state["t0"]
            self.cap_span = state["cap_span"]
            self.cap_n = state["cap_n"]
            self.net.P = state["P"]
        self._rebuild_tables()

    def _set_capacity(self):
        self.t0 = int(self.ts[0])
        self.cap_span = float(max(int(self.ts[-1]) - self.t0, 1)) * HEADROOM
        self.cap_n = int(self.ts.size * HEADROOM) + 1

    def _u(self, t):
        return (np.asarray(t, dtype=np.int64) - self.t0) / self.cap_span

    def _bucket(self, u):
        return np.clip((u * BUCKETS).astype(np.int64), 0, BUCKETS - 1)

    def _train(self, steps, lr):
        u = self._u(self.ts)
        y = np.arange(self.ts.size, dtype=np.float64) / self.cap_n
        self.net.fit(u, y, steps=steps, lr=lr)

    def _rebuild_tables(self):
        """Bucket boundaries and per-bucket maximum prediction error."""
        n = self.ts.size
        u = self._u(self.ts)
        bid = self._bucket(u)
        self.bstart = np.searchsorted(bid, np.arange(BUCKETS + 1)).astype(np.int64)

        pred = np.rint(self.net.predict(u) * self.cap_n).astype(np.int64)
        resid = np.abs(pred - np.arange(n, dtype=np.int64))
        nonempty = np.diff(self.bstart) > 0
        self.berr = np.zeros(BUCKETS, dtype=np.int64)
        self.berr[nonempty] = np.maximum.reduceat(resid, self.bstart[:-1][nonempty])

    def lower_bound(self, t):
        """
        For each timestamp in t: index of the first line with timestamp >= t.
        Returns (indices, window sizes searched, number of full-file fallbacks).
        """
        t = np.asarray(t, dtype=np.int64)
        u = self._u(t)
        b = self._bucket(u)
        s, e = self.bstart[b], self.bstart[b + 1]
        pred = np.rint(self.net.predict(u) * self.cap_n).astype(np.int64)
        err = self.berr[b]
        ts, n = self.ts, self.ts.size

        out = np.empty(t.size, np.int64)
        win = np.empty(t.size, np.int64)
        fallbacks = 0
        for q in range(t.size):
            tq = t[q]
            lo = min(max(max(int(s[q]), int(pred[q] - err[q] - 1)), 0), n)
            hi = min(max(min(int(e[q]), int(pred[q] + err[q] + 2)), lo), n)
            i = _bsearch(ts, tq, lo, hi)
            # Verify: i must be the exact lower bound. If not, search the whole file.
            if not ((i == n or ts[i] >= tq) and (i == 0 or ts[i - 1] < tq)):
                i = _bsearch(ts, tq, 0, n)
                fallbacks += 1
            out[q] = i
            win[q] = hi - lo
        return out, win, fallbacks

    def span(self, t_lo, t_hi):
        """Line range [a, b) with t_lo <= timestamp <= t_hi (inclusive)."""
        idx, _, _ = self.lower_bound(np.array([t_lo, t_hi + 1], dtype=np.int64))
        return int(idx[0]), int(idx[1])

    def read_lines(self, log_path, a, b):
        """Lines a..b-1, read with one seek and one read of their byte range."""
        lo = int(self.offsets[a]) if a < self.offsets.size else self.size
        hi = int(self.offsets[b]) if b < self.offsets.size else self.size
        with open(log_path, "rb") as f:
            f.seek(lo)
            blob = f.read(hi - lo)
        raw_lines = blob.split(b"\n")
        if raw_lines and raw_lines[-1] == b"":
            raw_lines.pop()
        lines = [ln.rstrip(b"\r") for ln in raw_lines]
        if len(lines) != b - a:
            raise RuntimeError("index does not match the log; run 'build' again")
        return lines, hi - lo

    def append(self, log_path, new_path, steps=300):
        """Appends lines to the log, then fine-tunes the model and refreshes the buckets."""
        new = np.fromfile(new_path, dtype=np.uint8)
        if new.size == 0:
            return 0
        if new[-1] != 10:
            new = np.append(new, np.uint8(10))
        ts_new, off_new, err_new = parse_log_bytes(new, base_offset=self.size)
        if ts_new[0] < self.ts[-1]:
            raise ValueError("appended lines must not be older than the log")
        with open(log_path, "ab") as f:
            f.write(new.tobytes())

        self.ts = np.concatenate([self.ts, ts_new])
        self.offsets = np.concatenate([self.offsets, off_new])
        self.is_err = np.concatenate([self.is_err, err_new])
        self.size += new.size

        if self.ts.size > self.cap_n or (self.ts[-1] - self.t0) >= self.cap_span:
            self._set_capacity()                 # log outgrew its capacity: new mapping
            self._train(steps=800, lr=0.005)     # warm start from the current weights
        else:
            self._train(steps=steps, lr=0.002)
        self._rebuild_tables()
        return ts_new.size

    def save(self, idx_path, log_path):
        np.savez(
            idx_path,
            ts=self.ts, offsets=self.offsets, is_err=self.is_err, size=self.size,
            t0=self.t0, cap_span=self.cap_span, cap_n=self.cap_n,
            log_mtime=int(os.path.getmtime(log_path)),
            **{"p_" + k: v for k, v in self.net.P.items()},
        )


def index_path(log_path):
    return log_path + ".loglearn.npz"


def open_index(log_path, rebuild=False):
    idx = index_path(log_path)
    if not rebuild and os.path.exists(idx):
        z = np.load(idx)
        if (int(z["size"]) == os.path.getsize(log_path)
                and int(z["log_mtime"]) == int(os.path.getmtime(log_path))):
            state = {
                "t0": int(z["t0"]),
                "cap_span": float(z["cap_span"]),
                "cap_n": int(z["cap_n"]),
                "P": {k: z["p_" + k].copy() for k in ("W1", "b1", "W2", "b2")},
            }
            return LogSeekIndex(z["ts"], z["offsets"], z["is_err"], int(z["size"]), state=state)

    ts, offs, err, size = parse_log_file(log_path)
    ix = LogSeekIndex(ts, offs, err, size)
    ix.save(idx, log_path)
    return ix


# =====================================================================
# 4. COMMANDS
# =====================================================================

def describe(ix):
    n = ix.ts.size
    nonempty = int(np.count_nonzero(np.diff(ix.bstart)))
    worst = int(ix.berr.argmax())
    model_kb = sum(v.nbytes for v in ix.net.P.values()) / 1024
    table_kb = (ix.bstart.nbytes + ix.berr.nbytes) / 1024
    nz = ix.berr[ix.berr > 0]
    median = np.median(nz) if nz.size else 0
    print(f"Lines: {n:,} | file: {ix.size / 1e6:.1f} MB | non-empty buckets: {nonempty}")
    print(f"Model error per bucket: median {median:.0f} lines, "
          f"worst {ix.berr.max():,} lines (bucket {worst})")
    print(f"Index size: model {model_kb:.2f} KB + bucket table {table_kb:.1f} KB")


def demo(ix):
    rng = np.random.default_rng(SEED)
    n = ix.ts.size
    base = ix.ts[rng.integers(0, n, 3000)]
    probe = np.concatenate([base, base + 1, base - 1])       # hits and misses
    got, win, fb = ix.lower_bound(probe)
    want = np.searchsorted(ix.ts, probe, side="left")
    assert np.array_equal(got, want), "index returned a wrong line"

    describe(ix)
    print(f"\nExact check: {probe.size:,} lookups (hits and misses) match numpy.searchsorted")
    print(f"Lookups that needed the full-file fallback: {fb}")
    print(f"Average search window: {win.mean():.1f} lines "
          f"(search starts from a range of {n:,} lines)")


def query(ix, log_path, args):
    lo = clock_ms(args.date, args.start, 0)
    hi = clock_ms(args.date, args.end, 999)
    a, b = ix.span(lo, hi)
    lines, nbytes = ix.read_lines(log_path, a, b)
    if not args.all:
        keep = ix.is_err[a:b]
        lines = [ln for ln, k in zip(lines, keep) if k]

    what = "lines" if args.all else "ERROR lines"
    print(f"{len(lines):,} {what} from {args.start} to {args.end} on {args.date}")
    print(f"Read {nbytes / 1024:.1f} KB of {ix.size / 1e6:.1f} MB")
    for ln in lines[:20]:
        print(ln.decode("utf-8", "replace"))
    if len(lines) > 20:
        print(f"... {len(lines) - 20:,} more")


def clock_ms(date, hms, ms):
    return int(np.datetime64(f"{date}T{hms}.{ms:03d}", "ms").astype(np.int64))


def generate_synthetic_logs(log_path, append_path=None, n_main=200_000, n_append=20_000,
                            force=False):
    """Generates server.log (diurnal wave + bursts) and optional new_lines.log."""
    for p in [log_path] + ([append_path] if append_path else []):
        if os.path.exists(p) and not force:
            raise SystemExit(f"'{p}' already exists; pass --force to overwrite it")

    rng = np.random.default_rng(SEED)
    total = n_main + (n_append if append_path else 0)

    # Diurnal traffic rate (sinusoidal wave) with exponential noise
    # Average gap ~60 ms so 200k lines starting at 12:00:00 span past 15:00:00
    phase = np.linspace(0, 3 * np.pi, total)
    base_gaps = (60.0 + 40.0 * np.sin(phase)).astype(np.float64)
    noise = rng.exponential(scale=1.0, size=total)
    gaps = np.maximum(1, np.rint(base_gaps * noise)).astype(np.int64)

    # Five intense traffic bursts (gaps of 0-1 ms)
    for burst_center in np.linspace(20_000, total - 15_000, 5, dtype=int):
        gaps[burst_center: burst_center + 1500] = rng.integers(0, 2, size=1500)

    start_epoch = int(np.datetime64("2026-11-10T12:00:00.000", "ms").astype(np.int64))
    ts_ms = start_epoch + np.cumsum(gaps)
    dt_strs = ts_ms.astype("datetime64[ms]").astype("U23")
    dt_strs = np.char.replace(dt_strs, "T", " ")

    levels = rng.choice(["INFO ", "WARN ", "ERROR"], size=total, p=[0.82, 0.10, 0.08])
    msgs = [
        "GET /api/v1/auth 200 OK",
        "POST /api/v2/checkout 502 Bad Gateway",
        "DB connection pool timeout on replica-3",
        "TLS handshake completed in 4ms",
    ]

    def _write_slice(path, lo, hi):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i in range(lo, hi):
                m = msgs[i % len(msgs)]
                f.write(f"{dt_strs[i]} {levels[i]} [worker-{i % 8}] {m}\n")

    _write_slice(log_path, 0, n_main)
    print(f"Generated '{log_path}' with {n_main:,} lines "
          f"({os.path.getsize(log_path) / 1e6:.2f} MB)")
    if append_path:
        _write_slice(append_path, n_main, total)
        print(f"Generated '{append_path}' with {n_append:,} lines "
              f"({os.path.getsize(append_path) / 1e6:.2f} MB)")


def run_paper_benchmark(log_path):
    """Generates experimental plots and metrics for the research paper."""
    import matplotlib.pyplot as plt

    ix = open_index(log_path, rebuild=True)
    demo(ix)

    # 1. Training convergence vs search window size
    step_checkpoints = [50, 200, 500, 1000, 1500]
    median_errs, mean_wins = [], []
    rng = np.random.default_rng(SEED)
    probe = ix.ts[rng.integers(0, ix.ts.size, 5000)]

    for st in step_checkpoints:
        sub_ix = LogSeekIndex(ix.ts, ix.offsets, ix.is_err, ix.size, steps=st)
        _, win, _ = sub_ix.lower_bound(probe)
        nz = sub_ix.berr[sub_ix.berr > 0]
        median_errs.append(float(np.median(nz)) if nz.size else 0.0)
        mean_wins.append(float(win.mean()))

    # 2. Three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: learned position curve vs true line positions
    sample_idx = np.linspace(0, ix.ts.size - 1, 1000, dtype=int)
    u_samp = ix._u(ix.ts[sample_idx])
    pred_pos = ix.net.predict(u_samp) * ix.cap_n
    axes[0].plot(u_samp, sample_idx, label="Actual line position", lw=2.2)
    axes[0].plot(u_samp, pred_pos, "--", label="TimeNet prediction (16 tanh units)", lw=2.0)
    axes[0].set_xlabel("Normalized timestamp (u)")
    axes[0].set_ylabel("Log line index")
    axes[0].set_title("(a) Neural fit on diurnal + burst logs")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: training steps vs search window
    axes[1].plot(step_checkpoints, mean_wins, marker="o", label="Avg search window (lines)")
    axes[1].plot(step_checkpoints, median_errs, marker="s", label="Median bucket error")
    axes[1].set_xlabel("Adam optimization steps")
    axes[1].set_ylabel("Number of log lines")
    axes[1].set_title("(b) Search window vs. training")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: bucket error distribution
    nonempty_errs = ix.berr[ix.berr > 0]
    axes[2].hist(nonempty_errs, bins=35, edgecolor="black", alpha=0.8)
    axes[2].set_xlabel("Max prediction error in bucket (lines)")
    axes[2].set_ylabel("Number of time buckets")
    axes[2].set_title("(c) Per-bucket error distribution")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("LogSeek experimental evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("logseek_results.png", dpi=150)
    print("\nSaved figure to 'logseek_results.png'")


def main():
    ap = argparse.ArgumentParser(description="LogSeek: learned time index for server logs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("log")
    g.add_argument("--append-out", default="new_lines.log")
    g.add_argument("--force", action="store_true", help="overwrite existing files")

    sub.add_parser("build").add_argument("log")
    sub.add_parser("demo").add_argument("log")
    sub.add_parser("benchmark").add_argument("log")

    sp = sub.add_parser("append")
    sp.add_argument("log")
    sp.add_argument("new")

    q = sub.add_parser("query")
    q.add_argument("log")
    q.add_argument("--date", required=True)
    q.add_argument("--start", required=True)
    q.add_argument("--end", required=True)
    q.add_argument("--all", action="store_true", help="include non-ERROR lines")
    args = ap.parse_args()

    if args.cmd == "generate":
        generate_synthetic_logs(args.log, args.append_out, force=args.force)
        return
    if args.cmd == "benchmark":
        run_paper_benchmark(args.log)
        return

    t0 = time.perf_counter()
    ix = open_index(args.log, rebuild=(args.cmd == "build"))
    print(f"Index ready in {time.perf_counter() - t0:.1f} s")

    if args.cmd == "build":
        describe(ix)
    elif args.cmd == "demo":
        demo(ix)
    elif args.cmd == "append":
        added = ix.append(args.log, args.new)
        ix.save(index_path(args.log), args.log)
        print(f"Appended {added:,} lines; model fine-tuned, buckets refreshed")
        describe(ix)
    elif args.cmd == "query":
        query(ix, args.log, args)


if __name__ == "__main__":
    main()