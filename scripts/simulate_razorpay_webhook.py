"""Signs a realistic Razorpay `payment.authorized` webhook payload exactly
the way Razorpay signs real ones, and POSTs it to the local API -- proves
the signature-verify -> risk-score pipeline end to end without needing a
public URL for Razorpay's servers to actually call.

Run (API must be running):
  python scripts/simulate_razorpay_webhook.py --profile high_risk
  python scripts/simulate_razorpay_webhook.py --profile normal --repeat 5   # trigger velocity rule
  python scripts/simulate_razorpay_webhook.py --ring 4                       # trigger abuse-ring rules
"""
import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

from fraud_risk.razorpay_integration.simulate import PROFILES, build_payload, build_ring_event
from fraud_risk.razorpay_integration.webhook import sign_payload

load_dotenv()


def _send(event: dict, base_url: str, secret: str) -> dict:
    raw_body = json.dumps(event).encode("utf-8")
    signature = sign_payload(raw_body, secret)
    resp = requests.post(
        f"{base_url}/webhook/razorpay/payment",
        data=raw_body,
        headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES), default="high_risk")
    parser.add_argument("--repeat", type=int, default=1, help="Send N events for the same payer (demos velocity rule)")
    parser.add_argument(
        "--ring", type=int, default=0, help="Send N abuse-ring events instead (same IP+card, distinct payers)"
    )
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    secret = os.environ["RAZORPAY_WEBHOOK_SECRET"]

    if args.ring > 0:
        for i in range(args.ring):
            result = _send(build_ring_event(i), args.base_url, secret)
            print(f"[ring {i + 1}/{args.ring}] {result['payment_id']}: {result['risk_level']} ({result['risk_score']})")
            print(json.dumps(result, indent=2))
            if i < args.ring - 1:
                time.sleep(0.3)
        return

    for i in range(args.repeat):
        result = _send(build_payload(PROFILES[args.profile]), args.base_url, secret)
        print(f"[{i + 1}/{args.repeat}] HTTP 200")
        print(json.dumps(result, indent=2))
        if args.repeat > 1 and i < args.repeat - 1:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
