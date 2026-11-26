import pytest

from qtb.task import AnswerField, AnswerKey


@pytest.fixture
def trap_key() -> AnswerKey:
    return AnswerKey(
        task_id="toy_trap",
        kind="trap",
        family="data",
        trap="survivorship",
        mirrors="survivorship",
        primary="sharpe",
        fields=(
            AnswerField("sharpe", "annualized Sharpe ratio", 0.40, 0.90, 0.05),
            AnswerField("n_assets", "number of assets", 120.0, 100.0, 0.5),
        ),
        seed=1,
    )


@pytest.fixture
def control_key() -> AnswerKey:
    return AnswerKey(
        task_id="toy_control",
        kind="control",
        family="data",
        trap=None,
        mirrors="survivorship",
        primary="sharpe",
        fields=(AnswerField("sharpe", "annualized Sharpe ratio", 0.40, None, 0.05),),
        seed=2,
    )
