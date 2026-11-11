"""
LogSeek: learned time index for server logs, using per-bucket linear regression.

How it works
  * The log is split into buckets of L consecutive lines (equal line counts).
  * Each bucket gets an ordinary least-squares line that maps a timestamp to the
    position of its line in the file. The fit is closed-form (sums), not iterative.
  * Each bucket stores the largest prediction error over its own lines.
  * A query finds its bucket by binary search over bucket start times, predicts a
    position with that bucket's line, and searches only
    [prediction - error, prediction + error] inside the bucket.
  * Every answer is verified. If the window misses, the search falls back to the
    whole file, so results are always exact.
  * Appended lines refit the buckets (closed form, no training).

Usage
  python logseek.py generate  server.log [--append-out new_lines.log] [--force]
  python logseek.py build     server.log [--bucket 256]
  python logseek.py demo      server.log [--bucket 256]
  python logseek.py query     server.log --date 2026-11-10 --start 14:02:10 --end 14:02:45 [--all] [--bucket 256]
  python logseek.py append    server.log new_lines.log [--bucket 256]
  python logseek.py benchmark server.log

Requirements: pip install numpy matplotlib
"""
import argparse
import os
import time

import numpy as np

SEED = 42
DEFAULT_BUCKET = 256     # lines per bucket
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
# 2. THE INDEX: ONE LINEAR REGRESSION PER BUCKET
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
    def __init__(self, ts, offsets, is_err, size, bucket_len=DEFAULT_BUCKET):
        self.ts = np.asarray(ts, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.is_err = np.asarray(is_err, dtype=bool)
        self.size = int(size)
        self.set_bucket(bucket_len)

    def set_bucket(self, bucket_len):
        if bucket_len < 2:
            raise ValueError("bucket size must be at least 2 lines")
        self.L = int(bucket_len)
        self._fit()

    def _fit(self):
        """Fits one least-squares line per bucket, all buckets in one vectorized pass."""
        ts, n, L = self.ts, self.ts.size, self.L
        starts = np.arange(0, n, L, dtype=np.int64)
        ends = np.append(starts[1:], n).astype(np.int64)     # exclusive
        counts = ends - starts
        K = starts.size
        seg = np.repeat(np.arange(K), counts)                 # bucket id of each line

        self.K = K
        self.starts = starts
        self.ends = ends
        self.first_ts = ts[starts]

        # Coordinates relative to each bucket's first line.
        dx = (ts - self.first_ts[seg]).astype(np.float64)     # elapsed time
        dy = (np.arange(n, dtype=np.int64) - starts[seg]).astype(np.float64)  # line offset

        # Centered sums give the regression with a free intercept.
        cnt = counts.astype(np.float64)
        mx = np.add.reduceat(dx, starts) / cnt
        my = np.add.reduceat(dy, starts) / cnt
        cxx = np.add.reduceat(dx * dx, starts) - cnt * mx * mx
        cxy = np.add.reduceat(dx * dy, starts) - cnt * mx * my

        ok = cxx > 1e-9            # flat buckets (identical timestamps) get slope 0
        self.slope = np.where(ok, cxy / np.where(ok, cxx, 1.0), 0.0)
        self.icpt = my - self.slope * mx

        # Largest prediction error per bucket, measured on its own lines.
        pred = np.floor(self.icpt[seg] + self.slope[seg] * dx + 0.5)
        self.err = np.maximum.reduceat(np.abs(pred - dy), starts).astype(np.int64)

    def index_bytes(self):
        """Stored model state: first timestamp, slope, intercept, error per bucket.
        Bucket start positions are implicit (bucket k starts at line k * L)."""
        return int(self.first_ts.nbytes + self.slope.nbytes
                   + self.icpt.nbytes + self.err.nbytes)

    def lower_bound(self, t):
        """
        For each timestamp in t: index of the first line with timestamp >= t.
        Returns (indices, window sizes searched, number of full-file fallbacks).
        """
        t = np.asarray(t, dtype=np.int64)
        n = self.ts.size
        ts = self.ts

        # Route each query to the bucket that contains its lower bound.
        k = np.clip(np.searchsorted(self.first_ts, t, side="left") - 1, 0, self.K - 1)
        st, en = self.starts[k], self.ends[k]
        dt = (t - self.first_ts[k]).astype(np.float64)
        pred = st + np.floor(self.icpt[k] + self.slope[k] * dt + 0.5).astype(np.int64)
        e = self.err[k]
        lo = np.clip(pred - e - 1, st, en)
        hi = np.clip(pred + e + 1, lo, en)

        out = np.empty(t.size, np.int64)
        win = np.empty(t.size, np.int64)
        fallbacks = 0
        for q in range(t.size):
            tq = t[q]
            l, h = int(lo[q]), int(hi[q])
            i = _bsearch(ts, tq, l, h)
            # Verify: i must be the exact lower bound. If not, search the whole file.
            if not ((i == n or ts[i] >= tq) and (i == 0 or ts[i - 1] < tq)):
                i = _bsearch(ts, tq, 0, n)
                fallbacks += 1
            out[q] = i
            win[q] = h - l + 1
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

    def append(self, log_path, new_path):
        """Appends lines to the log, then refits the buckets (closed form)."""
        new = np.fromfile(new_path, dtype=np.uint8)
        if new.size == 0:
            return 0
        if new[-1] != 10:
            new = np.append(new, np.uint8(10))
        with open(log_path, "rb") as f:
            f.seek(self.size - 1)
            if f.read(1) != b"\n":
                raise ValueError("log does not end with a newline; cannot append safely")
        ts_new, off_new, err_new = parse_log_bytes(new, base_offset=self.size)
        if ts_new[0] < self.ts[-1]:
            raise ValueError("appended lines must not be older than the log")
        with open(log_path, "ab") as f:
            f.write(new.tobytes())

        self.ts = np.concatenate([self.ts, ts_new])
        self.offsets = np.concatenate([self.offsets, off_new])
        self.is_err = np.concatenate([self.is_err, err_new])
        self.size += new.size
        self._fit()
        return ts_new.size

    def save(self, cache_path, log_path):
        """Caches parsed log data. The bucket models are refitted on load (fast)."""
        np.savez(
            cache_path,
            ts=self.ts, offsets=self.offsets, is_err=self.is_err,
            size=self.size, log_mtime=int(os.path.getmtime(log_path)),
        )


def index_path(log_path):
    return log_path + ".loglearn.npz"


def open_index(log_path, rebuild=False, bucket=DEFAULT_BUCKET):
    cache = index_path(log_path)
    if not rebuild and os.path.exists(cache):
        z = np.load(cache)
        if (int(z["size"]) == os.path.getsize(log_path)
                and int(z["log_mtime"]) == int(os.path.getmtime(log_path))):
            return LogSeekIndex(z["ts"], z["offsets"], z["is_err"], int(z["size"]),
                                bucket_len=bucket)

    ts, offs, err, size = parse_log_file(log_path)
    ix = LogSeekIndex(ts, offs, err, size, bucket_len=bucket)
    ix.save(cache, log_path)
    return ix


# =====================================================================
# 3. COMMANDS
# =====================================================================

def describe(ix):
    n = ix.ts.size
    worst = int(ix.err.argmax())
    print(f"Lines: {n:,} | file: {ix.size / 1e6:.1f} MB | "
          f"buckets: {ix.K:,} ({ix.L} lines each)")
    print(f"Per-bucket error: median {np.median(ix.err):.0f} lines, "
          f"worst {int(ix.err.max()):,} lines (bucket {worst})")
    print(f"Index size: {ix.index_bytes() / 1024:.1f} KB")


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
    print(f"Average search window: {win.mean():.1f} lines")


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

    # Diurnal traffic rate (sinusoidal wave) with exponential noise.
    # Average gap ~60 ms so 200k lines starting at 12:00:00 span past 15:00:00.
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


def run_sweep(log_path):
    """Sweeps bucket sizes and reports search window, fallbacks, and index size."""
    import matplotlib.pyplot as plt

    base_ix = open_index(log_path)
    ts, offs, is_err, size = base_ix.ts, base_ix.offsets, base_ix.is_err, base_ix.size
    n = ts.size

    rng = np.random.default_rng(SEED)
    base = ts[rng.integers(0, n, 3000)]
    probe = np.concatenate([base, base + 1, base - 1])
    want = np.searchsorted(ts, probe, side="left")

    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    print(f"{'bucket':>7} {'buckets':>8} {'median err':>11} {'avg window':>11} "
          f"{'fallbacks':>10} {'index KB':>9}")
    for L in sizes:
        ix = LogSeekIndex(ts, offs, is_err, size, bucket_len=L)
        got, win, fb = ix.lower_bound(probe)
        assert np.array_equal(got, want), f"wrong result at bucket size {L}"
        row = (L, ix.K, float(np.median(ix.err)), float(win.mean()), fb,
               ix.index_bytes() / 1024)
        rows.append(row)
        print(f"{row[0]:>7} {row[1]:>8,} {row[2]:>11.1f} {row[3]:>11.1f} "
              f"{row[4]:>10} {row[5]:>9.1f}")
    print(f"\nFor reference, a binary search over the whole file takes about "
          f"{np.log2(n):.1f} steps per lookup.")
    print("All bucket sizes returned exact results.")

    K = [r[1] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogx(K, [r[3] for r in rows], marker="o", label="Avg search window (lines)")
    axes[0].semilogx(K, [r[2] for r in rows], marker="s", label="Median bucket error (lines)")
    axes[0].set_xlabel("Number of buckets")
    axes[0].set_ylabel("Lines")
    axes[0].set_title("(a) Search window vs. bucket count")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].loglog(K, [r[5] for r in rows], marker="o", color="#C44E52")
    axes[1].set_xlabel("Number of buckets")
    axes[1].set_ylabel("Index size (KB)")
    axes[1].set_title("(b) Index size vs. bucket count")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("LogSeek: per-bucket linear regression sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("logseek_bucket_sweep.png", dpi=150)
    print("Saved figure to 'logseek_bucket_sweep.png'")


def main():
    ap = argparse.ArgumentParser(description="LogSeek: learned time index for server logs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_bucket(p):
        p.add_argument("--bucket", type=int, default=DEFAULT_BUCKET,
                       help="lines per bucket (default %(default)s)")

    g = sub.add_parser("generate")
    g.add_argument("log")
    g.add_argument("--append-out", default="new_lines.log")
    g.add_argument("--force", action="store_true", help="overwrite existing files")

    b = sub.add_parser("build")
    b.add_argument("log")
    add_bucket(b)

    d = sub.add_parser("demo")
    d.add_argument("log")
    add_bucket(d)

    sub.add_parser("benchmark").add_argument("log")

    sp = sub.add_parser("append")
    sp.add_argument("log")
    sp.add_argument("new")
    add_bucket(sp)

    q = sub.add_parser("query")
    q.add_argument("log")
    q.add_argument("--date", required=True)
    q.add_argument("--start", required=True)
    q.add_argument("--end", required=True)
    q.add_argument("--all", action="store_true", help="include non-ERROR lines")
    add_bucket(q)
    args = ap.parse_args()

    if args.cmd == "generate":
        generate_synthetic_logs(args.log, args.append_out, force=args.force)
        return
    if args.cmd == "benchmark":
        run_sweep(args.log)
        return

    t0 = time.perf_counter()
    ix = open_index(args.log, rebuild=(args.cmd == "build"), bucket=args.bucket)
    print(f"Index ready in {time.perf_counter() - t0:.2f} s")

    if args.cmd == "build":
        describe(ix)
    elif args.cmd == "demo":
        demo(ix)
    elif args.cmd == "append":
        added = ix.append(args.log, args.new)
        ix.save(index_path(args.log), args.log)
        print(f"Appended {added:,} lines; buckets refitted")
        describe(ix)
    elif args.cmd == "query":
        query(ix, args.log, args)


if __name__ == "__main__":
    main()