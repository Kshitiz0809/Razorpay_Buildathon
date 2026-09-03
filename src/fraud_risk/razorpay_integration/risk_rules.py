"""Stage-1 heuristic risk engine for live Razorpay payment events.

Unlike the calibrated LightGBM model in fraud_risk.models (trained and
measured on the labeled Kaggle benchmark), this engine scores real Razorpay
payment fields with no labeled ground truth available -- there is no
fraud/chargeback outcome history for a freshly-connected payment stream to
train or calibrate against. This is a deliberate, disclosed "cold start"
strategy: ship transparent, auditable rules on day one, instrument real
outcomes, then graduate to a supervised model (the same methodology already
proven on the benchmark dataset in fraud_risk.models) once enough labeled
outcomes exist for this account. See README.md.
"""
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class TriggeredRule:
    rule: str
    points: int
    reason: str


@dataclass
class RiskResult:
    risk_score: int
    risk_level: str
    triggered_rules: list[TriggeredRule]


class PaymentRiskEngine:
    """Stateful (in-process) so it can track simple velocity across events.

    In-memory state is a deliberate prototype simplification -- a real
    deployment would back this with a shared store (e.g. Redis) so it
    survives restarts and works across multiple API instances.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._history: dict[str, list[float]] = defaultdict(list)
        self._seen_payers: set[str] = set()
        self._ip_history: dict[str, list[float]] = defaultdict(list)
        self._card_payers: dict[tuple, set[str]] = defaultdict(set)

    def score(self, payment: dict) -> RiskResult:
        cfg = self.cfg
        triggered: list[TriggeredRule] = []

        divisor = cfg.get("currency_minor_unit_divisor", 100)
        amount_major = payment.get("amount", 0) / divisor
        email = payment.get("email") or ""
        contact = payment.get("contact") or ""
        method = payment.get("method") or ""
        card = payment.get("card") or {}
        created_at = payment.get("created_at", time.time())

        amt_cfg = cfg["amount"]
        if amount_major >= amt_cfg["very_high_amount_threshold"]:
            triggered.append(
                TriggeredRule(
                    "very_high_amount",
                    amt_cfg["very_high_amount_points"],
                    f"Amount {amount_major:,.0f} >= very-high threshold {amt_cfg['very_high_amount_threshold']:,.0f}",
                )
            )
        elif amount_major >= amt_cfg["high_amount_threshold"]:
            triggered.append(
                TriggeredRule(
                    "high_amount",
                    amt_cfg["high_amount_points"],
                    f"Amount {amount_major:,.0f} >= high threshold {amt_cfg['high_amount_threshold']:,.0f}",
                )
            )

        if card.get("international"):
            triggered.append(
                TriggeredRule("international_card", cfg["international_card"]["points"], "Card flagged international")
            )

        payer_key = email or contact
        if payer_key:
            window = cfg["velocity"]["window_seconds"]
            self._history[payer_key] = [t for t in self._history[payer_key] if created_at - t <= window] + [created_at]
            attempts = len(self._history[payer_key])
            if attempts > cfg["velocity"]["max_attempts_before_flag"]:
                triggered.append(
                    TriggeredRule(
                        "velocity",
                        cfg["velocity"]["points"],
                        f"{attempts} payment attempts from {payer_key} within {window}s",
                    )
                )

            if payer_key not in self._seen_payers:
                self._seen_payers.add(payer_key)
                if amount_major >= cfg["new_payer"]["amount_threshold"]:
                    triggered.append(
                        TriggeredRule(
                            "new_payer_high_amount",
                            cfg["new_payer"]["points"],
                            f"First-seen payer {payer_key} with amount {amount_major:,.0f}",
                        )
                    )

        # Abuse-ring signals: a coordinated ring probing with one device/card
        # against many synthetic identities shows up as either many payment
        # attempts from one IP, or one card fingerprint reused across
        # distinct payers -- neither is visible from a single transaction's
        # amount/method alone, which is why these need cross-event state.
        ip_address = (payment.get("notes") or {}).get("ip_address")
        if ip_address:
            ip_cfg = cfg["ip_velocity"]
            ip_window = ip_cfg["window_seconds"]
            self._ip_history[ip_address] = [
                t for t in self._ip_history[ip_address] if created_at - t <= ip_window
            ] + [created_at]
            ip_attempts = len(self._ip_history[ip_address])
            if ip_attempts > ip_cfg["max_attempts_before_flag"]:
                triggered.append(
                    TriggeredRule(
                        "ip_velocity",
                        ip_cfg["points"],
                        f"{ip_attempts} payment attempts from IP {ip_address} within {ip_window}s",
                    )
                )

        if card and payer_key:
            # last4+network+issuer is an imperfect card fingerprint (the
            # real card number/BIN is never exposed by Razorpay's API) but
            # is the standard imprecise signal real fraud systems use when
            # the full PAN isn't available.
            card_fp = (card.get("last4"), card.get("network"), card.get("issuer"))
            payers_for_card = self._card_payers[card_fp]
            payers_for_card.add(payer_key)
            if len(payers_for_card) > 1:
                triggered.append(
                    TriggeredRule(
                        "card_fingerprint_reuse",
                        cfg["card_fingerprint_reuse"]["points"],
                        f"Card ...{card_fp[0]} ({card_fp[1]}) used by {len(payers_for_card)} distinct payers",
                    )
                )

        method_points = cfg["method_risk_points"].get(method, 0)
        if method_points:
            triggered.append(
                TriggeredRule("method_risk", method_points, f"Payment method '{method}' baseline risk weight")
            )

        hour_ist = datetime.fromtimestamp(created_at, tz=IST).hour
        oh = cfg["odd_hour"]
        if oh["start_hour_ist"] <= hour_ist < oh["end_hour_ist"]:
            triggered.append(TriggeredRule("odd_hour", oh["points"], f"Transaction at {hour_ist:02d}:00 IST"))

        risk_score = min(100, sum(r.points for r in triggered))
        thresholds = cfg["decision_thresholds"]
        if risk_score >= thresholds["high"]:
            risk_level = "high"
        elif risk_score >= thresholds["medium"]:
            risk_level = "medium"
        else:
            risk_level = "low"

        return RiskResult(risk_score=risk_score, risk_level=risk_level, triggered_rules=triggered)
