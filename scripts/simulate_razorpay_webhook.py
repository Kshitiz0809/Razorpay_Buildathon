"""Signs a realistic Razorpay `payment.authorized` webhook payload exactly
the way Razorpay signs real ones, and POSTs it to the local API -- proves
the signature-verify -> risk-score pipeline end to end without needing a
public URL for Razorpay's servers to actually call.

Run (API must be running):
  python scripts/simulate_razorpay_webhook.py --profile high_risk
  python scripts/simulate_razorpay_webhook.py --profile normal --repeat 5   # trigger velocity rule
"""
import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

from fraud_risk.razorpay_integration.simulate import PROFILES, build_payload
from fraud_risk.razorpay_integration.webhook import sign_payload

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES), default="high_risk")
    parser.add_argument("--repeat", type=int, default=1, help="Send N events for the same payer (demos velocity rule)")
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    secret = os.environ["RAZORPAY_WEBHOOK_SECRET"]

    for i in range(args.repeat):
        event = build_payload(PROFILES[args.profile])
        raw_body = json.dumps(event).encode("utf-8")
        signature = sign_payload(raw_body, secret)

        resp = requests.post(
            f"{args.base_url}/webhook/razorpay/payment",
            data=raw_body,
            headers={"Content-Type": "application/json", "X-Razorpay-Signature": signature},
            timeout=10,
        )
        print(f"[{i + 1}/{args.repeat}] HTTP {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
        if args.repeat > 1 and i < args.repeat - 1:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
