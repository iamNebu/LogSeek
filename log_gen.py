"""
Generate a synthetic server log with steady traffic and periodic bursts.

Line format (fixed-width timestamp at the start of each line):
    2026-11-10 14:02:10.123 INFO GET /api/items 200

Usage:
    python generate_log.py                      # 2,000,000 lines, ~200 MB
    python generate_log.py --lines 20000000     # larger file
"""
import argparse
import datetime as dt
import random

BASE_TIME = dt.datetime(2026, 11, 10, 14, 0, 0)
LEVELS = ["INFO"] * 90 + ["WARN"] * 7 + ["ERROR"] * 3
REQUESTS = [
    "GET /api/items 200",
    "POST /login 200",
    "GET /static/app.js 304",
    "GET /api/orders 500",
    "GET /health 200",
]


def generate(path, n_lines, seed=42,
             normal_rate=50.0,        # lines per second, steady traffic
             burst_rate=5000.0,       # lines per second during a burst
             burst_every=500_000,     # a burst starts every N lines
             burst_len=20_000):       # burst length in lines
    rng = random.Random(seed)
    t = 0.0                           # seconds since BASE_TIME
    with open(path, "w", buffering=1 << 20) as f:
        for i in range(n_lines):
            in_burst = (i % burst_every) < burst_len
            rate = burst_rate if in_burst else normal_rate
            t += rng.expovariate(rate)
            stamp = BASE_TIME + dt.timedelta(seconds=t)
            ts = f"{stamp:%Y-%m-%d %H:%M:%S}.{stamp.microsecond // 1000:03d}"
            level = rng.choice(LEVELS)
            req = rng.choice(REQUESTS)
            f.write(f"{ts} {level} {req}\n")
    print(f"Wrote {n_lines:,} lines to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="server.log")
    p.add_argument("--lines", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate(args.out, args.lines, seed=args.seed)