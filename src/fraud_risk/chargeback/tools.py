"""Read-only investigation tools for the chargeback agent.

Three tools, exposed to the LLM via OpenAI-compatible function-calling
schemas. None of them write anything -- the one action tool (submitting
evidence) lives separately in mock_submission.py, since it's a
fundamentally different kind of capability (a write, gated by the caller).
"""
import hashlib
import json

import requests

from fraud_risk.config import ROOT

CHARGEBACK_HISTORY_PATH = ROOT / "data" / "mock_services" / "chargeback_history.json"

# Illustrative keyword list for classifying an IP's ISP/org as a
# datacenter/hosting provider rather than a residential or business
# connection -- a well-documented fraud signal (VPNs, proxies, bot
# infrastructure often route through these). Not exhaustive; a production
# deployment would use a dedicated IP-intelligence provider (MaxMind,
# IPQualityScore, etc.) instead.
_DATACENTER_KEYWORDS = [
    "amazon", "aws", "google", "microsoft", "azure", "digitalocean",
    "linode", "ovh", "hetzner", "cloudflare", "vultr", "oracle cloud",
]


def check_ip_reputation(ip_address: str) -> dict:
    """Real lookup via ip-api.com (free, unauthenticated, no key needed),
    with a deterministic local fallback if that service is unreachable --
    so this tool always returns something instead of breaking a live
    investigation (or a live demo) on a third-party outage.
    """
    try:
        resp = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            org_text = f"{data.get('isp', '')} {data.get('org', '')}".lower()
            is_datacenter = any(kw in org_text for kw in _DATACENTER_KEYWORDS)
            return {
                "ip_address": ip_address,
                "source": "ip-api.com",
                "country": data.get("country"),
                "isp": data.get("isp"),
                "org": data.get("org"),
                "classification": "datacenter_or_hosting" if is_datacenter else "residential_or_business",
            }
    except requests.RequestException:
        pass

    digit_sum = sum(int(c) for c in ip_address if c.isdigit())
    is_datacenter = digit_sum % 5 == 0
    return {
        "ip_address": ip_address,
        "source": "local_heuristic_fallback (ip-api.com unreachable)",
        "country": "unknown",
        "isp": "unknown",
        "org": "unknown",
        "classification": "datacenter_or_hosting" if is_datacenter else "residential_or_business",
    }


def check_shipping_status(tracking_number: str) -> dict:
    """Simulated carrier lookup.

    No universal free carrier-tracking API exists without a per-carrier
    key, and this project's transaction dataset has no real order/shipping
    data to tie to -- so this is honestly a deterministic simulation
    (hashed on the tracking number, so the same input always returns the
    same status), not a live integration.
    """
    digest = int(hashlib.sha256(tracking_number.encode("utf-8")).hexdigest(), 16)
    statuses = ["delivered", "in_transit", "out_for_delivery", "delivery_exception"]
    status = statuses[digest % len(statuses)]
    return {
        "tracking_number": tracking_number,
        "source": "simulated (no live carrier integration in this prototype)",
        "status": status,
        "delivered": status == "delivered",
    }


def check_prior_chargebacks(email: str) -> dict:
    """Queries this project's own local chargeback-history store (seeded
    demo data at data/mock_services/chargeback_history.json, not real
    production records)."""
    if CHARGEBACK_HISTORY_PATH.exists():
        history = json.loads(CHARGEBACK_HISTORY_PATH.read_text())
    else:
        history = {}

    record = history.get(
        email, {"prior_chargebacks": 0, "account_age_days": None, "notes": "no history on file"}
    )
    return {"email": email, "source": "local chargeback history store", **record}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_ip_reputation",
            "description": (
                "Look up whether an IP address belongs to a residential/business connection "
                "or a known datacenter/hosting/VPN provider."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ip_address": {"type": "string", "description": "The customer's IP address at time of purchase"}
                },
                "required": ["ip_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_shipping_status",
            "description": "Check whether a shipment has been delivered, given its tracking number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string", "description": "The shipment tracking number for this order"}
                },
                "required": ["tracking_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_prior_chargebacks",
            "description": "Look up this customer's prior chargeback/dispute history with this merchant.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string", "description": "The customer's email address"}},
                "required": ["email"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "check_ip_reputation": check_ip_reputation,
    "check_shipping_status": check_shipping_status,
    "check_prior_chargebacks": check_prior_chargebacks,
}
