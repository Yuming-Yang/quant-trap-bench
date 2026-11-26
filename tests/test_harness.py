import json

from qtb.harness import AgentConfig, LLMAgent, opening_message
from qtb.models.base import FakeModel, ModelTurn, ToolCall, Usage, run_code, say
from qtb.runner import run_task
from qtb.scorer import Outcome

NAIVE_CODE = """
import json, pandas as pd
r = pd.read_csv('returns.csv')
w = r.pivot(index='date', columns='ticker', values='ret')
json.dump({'ew_mean': float(w.mean(axis=1).mean() * 12)}, open('answer.json', 'w'))
print('done')
"""
CAREFUL_CODE = NAIVE_CODE.replace(
    "r = pd.read_csv('returns.csv')",
    "r = pd.concat([pd.read_csv('returns.csv'), pd.read_csv('delisted_returns.csv')])",
)


def agent(script, max_steps=10, prompt="plain"):
    model = FakeModel(script)
    return LLMAgent(AgentConfig(model, prompt=prompt, max_steps=max_steps)), model


def test_a_scripted_model_that_takes_the_data_at_face_value_scores_naive(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    llm, _ = agent(
        [
            run_code("print(open('DATA_DICTIONARY.md').read())"),
            run_code(NAIVE_CODE),
            say("EW mean computed from returns.csv."),
        ]
    )
    result = run_task(llm, task)
    assert result.score.outcome is Outcome.NAIVE
    assert result.meta["stop"] == "end_turn"
    assert result.meta["steps"] == 3
    assert result.meta["tool_calls"] == 2
    assert result.meta["report"] == "EW mean computed from returns.csv."


def test_a_scripted_model_that_reads_the_dictionary_scores_corrected(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    llm, _ = agent([run_code(CAREFUL_CODE), say("Included the delisted stocks.")])
    assert run_task(llm, task).score.outcome is Outcome.CORRECTED


def test_step_limit_ends_the_run_and_a_missing_answer_scores_missing(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    llm, model = agent([run_code("print(1)", call_id=f"c{i}") for i in range(10)], max_steps=4)
    result = run_task(llm, task)
    assert result.meta["stop"] == "step_limit"
    assert result.meta["steps"] == 4
    assert len(model.requests) == 4
    assert result.score.outcome is Outcome.MISSING


def test_tokens_are_summed_over_steps(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    llm, _ = agent(
        [
            run_code("print(1)", usage=Usage(1000, 50, 800, 100)),
            say("done", usage=Usage(1200, 30, 1000, 150)),
        ]
    )
    usage = run_task(llm, task).meta["usage"]
    assert usage == {
        "input_tokens": 2200,
        "output_tokens": 80,
        "cache_read_input_tokens": 1800,
        "cache_creation_input_tokens": 250,
    }


def test_all_tool_results_go_back_in_one_message(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    two_calls = ModelTurn(
        [
            {"type": "tool_use", "id": "a", "name": "python", "input": {"code": "print('A')"}},
            {"type": "tool_use", "id": "b", "name": "python", "input": {"code": "1/0"}},
        ],
        "",
        [ToolCall("a", "python", {"code": "print('A')"}), ToolCall("b", "python", {"code": "1/0"})],
        "tool_use",
    )
    llm, model = agent([two_calls, say("ok")])
    result = run_task(llm, task)
    sent = model.requests[1]["messages"][-1]
    assert sent["role"] == "user"
    assert [b["tool_use_id"] for b in sent["content"]] == ["a", "b"]
    assert sent["content"][0]["content"].strip() == "A"
    assert sent["content"][1]["is_error"] is True
    assert "is_error" not in sent["content"][0]
    events = result.meta["events"]
    assert events[0]["tool_calls"][1]["exit_code"] == 1


def test_assistant_content_is_sent_back_unchanged(tasks_by_id):
    task = tasks_by_id["survivorship_1"]
    thinking = {"type": "thinking", "thinking": "", "signature": "opaque"}
    turn = run_code("print(1)")
    turn = ModelTurn([thinking, *turn.content], "", turn.tool_calls, "tool_use")
    llm, model = agent([turn, say("ok")])
    run_task(llm, task)
    assert model.requests[1]["messages"][1] == {"role": "assistant", "content": turn.content}


def test_refusal_ends_the_run(tasks_by_id):
    refusal = ModelTurn([], "", [], "refusal")
    llm, model = agent([refusal, say("never reached")])
    result = run_task(llm, tasks_by_id["survivorship_1"])
    assert result.meta["stop"] == "refusal"
    assert len(model.requests) == 1


def test_opening_message_has_request_dictionary_and_files_but_no_key(tasks_by_id, tmp_path):
    import shutil

    task = tasks_by_id["point_in_time_1"]
    workdir = tmp_path / "w"
    shutil.copytree(task.public_dir, workdir)
    text = opening_message(task, workdir)
    assert task.request.strip() in text
    assert "publish_date" in text
    assert "`fundamentals.csv`" in text and "`returns.csv`" in text
    key = json.loads((task.root / "key.json").read_text())
    assert f"{key['fields'][0]['naive']}" not in text
    assert "key.json" not in text


def test_prompts_do_not_name_traps():
    from qtb.harness import SYSTEM_PROMPTS

    for prompt in SYSTEM_PROMPTS.values():
        for word in ["trap", "survivorship", "lookahead", "bias", "point-in-time"]:
            assert word not in prompt.lower()


def test_code_that_reaches_for_the_key_is_flagged(tasks_by_id):
    llm, _ = agent([run_code("import os; print(os.path.exists('../key.json'))"), say("x")])
    assert run_task(llm, tasks_by_id["survivorship_1"]).meta["suspicious"] == ["key.json"]


def test_review_prompt_is_a_separate_configuration():
    llm, _ = agent([], prompt="review")
    assert llm.name == "fake+review"
