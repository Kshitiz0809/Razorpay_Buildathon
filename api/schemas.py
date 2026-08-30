from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, create_model


class _APIModel(BaseModel):
    # "model_version" is a deliberate, public field name; pydantic reserves
    # the "model_" prefix for its own internals by default and warns on any
    # collision, so opt out of that protection for this API's schemas.
    model_config = ConfigDict(protected_namespaces=())


# v1..v28 generated to avoid 28 hand-typed near-identical lines, but the
# resulting model still exposes explicit named fields (not a bare list) in
# the OpenAPI contract -- each gets its own validation error if malformed.
_V_FIELDS = {f"v{i}": (float, ...) for i in range(1, 29)}


class _TransactionBase(BaseModel):
    transaction_id: str | None = None
    time: float = Field(..., ge=0, description="Seconds since the reference epoch")
    amount: float = Field(..., ge=0)


TransactionIn = create_model("TransactionIn", __base__=_TransactionBase, **_V_FIELDS)


class ScoreOut(_APIModel):
    transaction_id: str | None
    fraud_probability: float
    is_flagged: bool
    threshold_used: float
    model_version: str
    scored_at: datetime


class FeatureContribution(BaseModel):
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str


class ExplainOut(ScoreOut):
    top_contributors: list[FeatureContribution]


class HealthOut(_APIModel):
    status: str
    model_version: str
