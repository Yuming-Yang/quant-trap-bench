import pytest

from qtb.cost import Scenario, scenario_usage, usage_cost


def test_usage_cost_by_hand():
    usage = {
        "input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    assert usage_cost(usage, "claude-opus-5-5") == pytest.approx(4 + 5 + 0.2 + 20)


def test_scenario_counts_a_growing_cached_prefix():
    s = Scenario("toy", steps=3, opening_tokens=100, output_per_step=10, tool_result_per_step=5)
    # Request 1 writes 100. Request 2 reads 100, writes 15. Request 3 reads 115, writes 15.
    assert scenario_usage(s) == {
        "input_tokens": 0,
        "cache_creation_input_tokens": 130,
        "cache_read_input_tokens": 215,
        "output_tokens": 30,
    }
