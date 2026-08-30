"""Creates a handful of real sample Orders in your Razorpay TEST-mode
account via the live Orders API -- safe and non-monetary (an Order only
reserves an amount; no payment is completed against it). Proves genuine
write access: the resulting order IDs are visible in your own Razorpay
Dashboard (Test Mode > Orders).

Refuses to run against a non-test key_id (see RazorpayClient.create_order).

Run:  python scripts/seed_razorpay_test_orders.py
"""
import uuid

from dotenv import load_dotenv

from fraud_risk.razorpay_integration.client import RazorpayClient

load_dotenv()

SAMPLE_ORDERS = [
    {"amount_minor": 149900, "note": "Sample order #1 -- typical checkout"},
    {"amount_minor": 499900, "note": "Sample order #2 -- higher-value checkout"},
    {"amount_minor": 29900, "note": "Sample order #3 -- small ticket item"},
]


def main():
    client = RazorpayClient()
    if not client.is_test_mode:
        raise SystemExit("Refusing to run: RAZORPAY_KEY_ID does not look like a test-mode key.")

    for sample in SAMPLE_ORDERS:
        order = client.create_order(
            amount_minor=sample["amount_minor"],
            currency="INR",
            receipt=f"demo-{uuid.uuid4().hex[:10]}",
            notes={"source": "fraud-risk-scorer-demo", "description": sample["note"]},
        )
        print(f"Created {order['id']}: INR {order['amount'] / 100:,.2f} -- {sample['note']}")

    print("\nDone. View these in your Razorpay Dashboard under Test Mode > Orders.")


if __name__ == "__main__":
    main()
