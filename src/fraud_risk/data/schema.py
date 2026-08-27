import pandera as pa
from pandera import Column, Check

from fraud_risk.config import V_COLUMNS

_v_columns = {v: Column(float, nullable=False) for v in V_COLUMNS}

RawTransactionSchema = pa.DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), nullable=False),
        "Amount": Column(float, Check.ge(0), nullable=False),
        **_v_columns,
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=True,
)


def validate_raw(df):
    return RawTransactionSchema.validate(df)
