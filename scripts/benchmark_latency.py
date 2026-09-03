"""Measures real /score latency against a running API instance.

Run the API first (uvicorn api.main:app), then:
  python scripts/benchmark_latency.py --n 200
"""
import argparse
import os
import statistics
import time

import requests


def sample_payload(i: int) -> dict:
    payload = {"transaction_id": f"bench_{i}", "time": 1000.0 + i, "amount": 50.0 + (i % 100)}
    payload.update({f"v{j}": ((i + j) % 10) / 5.0 - 1.0 for j in range(1, 29)})
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set API_KEY env var or pass --api-key")

    headers = {"X-API-Key": args.api_key}
    latencies_ms = []

    # A persistent Session reuses the underlying TCP connection across
    # requests (HTTP keep-alive) -- without it, requests.post() opens a
    # fresh connection per call, and on this machine that connection-setup
    # overhead dominated the measurement far more than actual scoring time.
    with requests.Session() as session:
        for i in range(args.n):
            start = time.perf_counter()
            resp = session.post(f"{args.base_url}/score", json=sample_payload(i), headers=headers, timeout=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            latencies_ms.append(elapsed_ms)
            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{args.n}", flush=True)

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p99 = latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.99))]

    print(f"n={args.n} requests")
    print(f"  mean : {statistics.mean(latencies_ms):.2f} ms")
    print(f"  p50  : {p50:.2f} ms")
    print(f"  p99  : {p99:.2f} ms")
    print(f"  max  : {max(latencies_ms):.2f} ms")


if __name__ == "__main__":
    main()
