"""Thin authenticated wrapper around Razorpay's REST API.

Read operations (list_payments, list_orders) and one safe write operation
(create_order, which only reserves an amount -- it does not move money; a
payment still has to be completed against it through Razorpay's Checkout
flow, which this backend deliberately cannot do server-side, by Razorpay's
own PCI-DSS design).
"""
import os

import requests

BASE_URL = "https://api.razorpay.com/v1"


class RazorpayClient:
    def __init__(self, key_id: str | None = None, key_secret: str | None = None):
        self.key_id = key_id or os.environ["RAZORPAY_KEY_ID"]
        self.key_secret = key_secret or os.environ["RAZORPAY_KEY_SECRET"]

    def _auth(self):
        return (self.key_id, self.key_secret)

    @property
    def is_test_mode(self) -> bool:
        return self.key_id.startswith("rzp_test_")

    def list_payments(self, count: int = 20) -> list[dict]:
        resp = requests.get(f"{BASE_URL}/payments", auth=self._auth(), params={"count": count}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("items", [])

    def list_orders(self, count: int = 20) -> list[dict]:
        resp = requests.get(f"{BASE_URL}/orders", auth=self._auth(), params={"count": count}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("items", [])

    def create_order(
        self,
        amount_minor: int,
        currency: str = "INR",
        receipt: str | None = None,
        notes: dict | None = None,
    ) -> dict:
        if not self.is_test_mode:
            raise RuntimeError(
                "Refusing to create an order: key_id does not start with 'rzp_test_'. "
                "This client only writes against test-mode accounts."
            )
        payload: dict = {"amount": amount_minor, "currency": currency}
        if receipt:
            payload["receipt"] = receipt
        if notes:
            payload["notes"] = notes
        resp = requests.post(f"{BASE_URL}/orders", auth=self._auth(), json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
