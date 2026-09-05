"""Tool-calling investigation agent for chargeback disputes.

The hard safety gate -- refusing to run at all on a transaction the fraud
model itself scored as high-risk -- lives in api/main.py and is enforced
BEFORE this module ever runs, deterministically, in code. That gate is
never delegated to the LLM. What the LLM decides, within that already-
permitted envelope, is which investigation tools to call, when it has
gathered enough evidence, and whether to submit -- the model does not
decide whether it is allowed to act, only how to act once permitted.
"""
import json
import os

from fraud_risk.chargeback.groq_client import DEFAULT_MODEL, call_groq
from fraud_risk.chargeback.mock_submission import SUBMIT_TOOL_SCHEMA, submit_dispute_evidence
from fraud_risk.chargeback.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MAX_TOOL_ITERATIONS = 6


def _call_groq(messages: list[dict], tools: list[dict], api_key: str, model: str) -> dict:
    return call_groq(
        api_key,
        {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            # The submit_dispute_evidence tool call's "letter" argument is
            # an entire JSON-escaped dispute letter -- 900 tokens (fine for
            # plain conversational replies, see responder.py) was not
            # enough headroom for that plus the model's own reasoning
            # overhead, and Groq truncated generation mid-argument, which
            # then failed to parse as JSON ("tool_use_failed") -- confirmed
            # live during development, not a hypothetical.
            "max_tokens": 2000,
            "reasoning_effort": "low",
        },
    )


def _format_contributors(top_contributors: list[dict]) -> str:
    return "\n".join(
        f"- {c['feature_name']}: {c['direction'].replace('_', ' ')} (contribution {c['shap_value']:+.3f})"
        for c in top_contributors
    )


def investigate_and_respond(
    transaction_id: str,
    amount: float,
    currency: str,
    transaction_date: str,
    merchant_name: str,
    top_contributors: list[dict],
    customer_email: str | None = None,
    customer_ip: str | None = None,
    tracking_number: str | None = None,
    allow_submit: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict:
    api_key = os.environ["GROQ_API_KEY"]

    available_tools = list(TOOL_SCHEMAS)
    tool_functions = dict(TOOL_FUNCTIONS)
    if allow_submit:
        available_tools = available_tools + [SUBMIT_TOOL_SCHEMA]
        tool_functions["submit_dispute_evidence"] = lambda letter: submit_dispute_evidence(transaction_id, letter)

    known_facts = [f"Transaction ID: {transaction_id}", f"Amount: {currency} {amount:,.2f}", f"Date: {transaction_date}"]
    if customer_email:
        known_facts.append(f"Customer email: {customer_email}")
    if customer_ip:
        known_facts.append(f"Customer IP at purchase: {customer_ip}")
    if tracking_number:
        known_facts.append(f"Shipment tracking number: {tracking_number}")

    action_instruction = (
        "Once you've gathered enough evidence, call submit_dispute_evidence with your final letter."
        if allow_submit
        else "You may NOT submit anything. End with your recommended letter text, clearly marked as a DRAFT for human review."
    )

    system_prompt = (
        f"You are a chargeback dispute investigator for {merchant_name}. A customer has filed a "
        "chargeback on a transaction our fraud model already scored as LOW risk (legitimate). Your "
        "job: investigate using the available tools, decide whether the evidence supports contesting "
        f"the chargeback, and {'draft and submit' if allow_submit else 'draft'} a dispute letter.\n\n"
        f"Known facts:\n{chr(10).join(known_facts)}\n\n"
        f"Our fraud model's risk signals for this transaction (SHAP explanation):\n"
        f"{_format_contributors(top_contributors)}\n\n"
        "Investigate using the tools available to you before concluding. Only use facts you actually "
        f"gathered -- never invent evidence. {action_instruction}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Begin your investigation."},
    ]

    trace: list[dict] = []
    submitted = False
    submission_reference = None
    final_text = ""

    for _ in range(MAX_TOOL_ITERATIONS):
        response = _call_groq(messages, available_tools, api_key, model)
        message = response["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            final_text = message.get("content") or ""
            break

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = json.loads(call["function"]["arguments"] or "{}")
            fn = tool_functions.get(fn_name)
            result = {"error": f"unknown tool {fn_name}"} if fn is None else fn(**fn_args)

            if fn_name == "submit_dispute_evidence" and isinstance(result, dict) and result.get("submitted"):
                submitted = True
                submission_reference = result.get("confirmation_id")

            trace.append({"tool": fn_name, "arguments": fn_args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    else:
        final_text = "Investigation did not conclude within the allotted steps."

    return {
        "trace": trace,
        "conclusion": final_text,
        "submitted": submitted,
        "submission_reference": submission_reference,
    }
