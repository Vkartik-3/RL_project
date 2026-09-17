import pytest
import torch

from forgeline.core.errors import InvalidRewardError
from forgeline.core.protocols import Trajectory
from forgeline.rollouts.engine import RolloutConfig, RolloutEngine, stack_group
from forgeline.rollouts.rewards import (
    CompositeReward, ConstantReward, CriticReward, LengthReward, ProcessReward, RepetitionReward, TextFormatReward,
    VerifierReward, build_reward_provider, compute_process_reward, heuristic_critic_score, parse_steps, score_step, score_trajectories,
)
from forgeline.rollouts.tools.parser import parse_tagged_output
from forgeline.rollouts.tools.python_executor import execute_python
from forgeline.rollouts.verifiers import (
    CodeExecutionVerifier, ExactMatchVerifier, FormatVerifier, MathVerifier, TaggedAnswerVerifier, build_verifier, tag_format_reward,
)


def traj(text: str, task=None) -> Trajectory:
    return Trajectory(prompt_ids=torch.tensor([1, 2]), response_ids=torch.tensor([3, 4, 5, 3, 4]), response_text=text, task=task or {})


def test_math_verifier():
    v = MathVerifier()
    assert v.verify("The answer is 42", "42").passed
    assert v.verify("so \\boxed{1,024}", "1024").value == 1.0
    assert v.verify("#### 12", "12").passed
    r = v.verify("about 100.5", "100")
    assert r.value == 0.5 and not r.passed
    assert v.verify("no numbers here", "3").value == 0.0
    assert not v.verify("x", None).passed and "error" in v.verify("x", None).info


def test_tagged_and_exact_and_format():
    t = TaggedAnswerVerifier()
    assert t.verify("<final_answer>5,200</final_answer>", "5200").value == 1.0
    assert t.verify("<tool_call>{}</tool_call><tool_result>1</tool_result>", "9").value == 0.1
    assert t.verify("nothing", "9").value == 0.0
    assert tag_format_reward("<think></think><tool_call></tool_call><final_answer>1</final_answer>") == pytest.approx(0.06)
    assert ExactMatchVerifier().verify("  Hello  World ", "hello world").passed
    assert FormatVerifier("cot").verify("<think>" + "x" * 30 + "</think> answer here").passed
    assert FormatVerifier("json").verify('{"a": 1}').value == 1.0
    assert FormatVerifier("steps").verify("Step 1: a\nStep 2: b").passed


def test_code_verifier_and_executor():
    assert execute_python("print(2+2)").stdout == "4"
    assert execute_python("import time; time.sleep(3)", timeout=0.5).timed_out
    v = CodeExecutionVerifier()
    assert v.verify("```python\ndef add(a, b):\n    return a + b\n```", "assert add(1, 2) == 3").passed
    assert v.verify("def add(a, b):\n    return a - b", "assert add(1, 2) == 3").value == 0.0
    assert v.verify("garbage(", "assert True").value == -0.1


def test_parser():
    p = parse_tagged_output('<think>t</think><tool_call>{"name": "python_executor", "args": {"code": "print(1)"}}</tool_call>')
    assert p.tool_call.name == "python_executor" and p.thinking == "t" and p.final_answer is None
    assert parse_tagged_output("<tool_call>not json</tool_call>").tool_call is None


def test_rule_rewards_and_composite():
    t = traj("Hello.\nworld")
    assert LengthReward(5).score(t).value == 1.0 and TextFormatReward().score(t).value == pytest.approx(0.8)
    assert 0 < RepetitionReward().score(t).value <= 1.0
    comp = CompositeReward([(ConstantReward(1.0), 0.7), (ConstantReward(0.5), 0.3)])
    assert comp.score(t).value == pytest.approx(0.85)
    with pytest.raises(InvalidRewardError):
        CompositeReward([(ConstantReward(float("nan")), 1.0)]).score(t)
    with pytest.raises(InvalidRewardError):
        score_trajectories(ConstantReward(float("inf")), [t])


def test_verifier_reward_and_builder():
    r = VerifierReward(MathVerifier(), "answer").score(traj("answer is 7", {"answer": "7"}))
    assert r.passed and r.value == 1.0
    p = build_reward_provider({"type": "composite", "providers": [{"type": "verifier", "verifier": "math", "weight": 0.7},
                                                                  {"type": "rule", "name": "constant", "value": 1.0, "weight": 0.3}]})
    assert p.score(traj("answer is 7", {"answer": "7"})).value == pytest.approx(1.0)
    assert build_reward_provider({"type": "tagged"}).score(traj("<final_answer>7</final_answer>", {"answer": "7"})).value == pytest.approx(1.02)
    assert isinstance(build_verifier("math"), MathVerifier)


def test_process_reward():
    resp = "Step 1: x = 3 * 24 = 72\n\nStep 2: y = 72 + 8\n\n<final_answer>80</final_answer>"
    total, scores = compute_process_reward(resp, "80", gamma=0.9, step_weight=1.0, execute_code=False)
    assert scores[-1] == 1.0 and scores[0] == 0.15 and total > 1.0
    assert score_step("x = 3 * 24 = 73") == 0.10 and score_step("just words") == 0.0 and score_step("12 + 5 more") == 0.10
    assert len(parse_steps("alpha paragraph with plenty of detail\n\nbeta paragraph with plenty of detail\n\nend")) == 2
    r = ProcessReward(gamma=0.9, step_weight=0.5, execute_code=False).score(traj(resp, {"answer": "80"}))
    assert r.passed and 0 < r.info["step_fraction"] < 1


def test_critic():
    out = "<think>first compute then calculate because</think><tool_call>x</tool_call><final_answer>1</final_answer> " + "word " * 20
    assert heuristic_critic_score("p", out) >= 0.75
    assert CriticReward("llm", critic_fn=lambda p: "Score: 0.8").score(traj("x")).value == 0.8
    with pytest.raises(ValueError):
        CriticReward("llm")


def test_rollout_engine_groups(policy):
    engine = RolloutEngine(policy, RolloutConfig(group_size=3, max_new_tokens=6, temperature=1.0))
    group = engine.rollout(torch.tensor(policy.encode("Q:")), {"answer": "1"}, prompt_text="Q:")
    assert len(group) == 3 and all(t.old_logprobs is not None and t.old_logprobs.shape[0] == t.response_length for t in group)
    prompt, resp, mask = stack_group(group)
    assert prompt.shape[0] == 3 and resp.shape == mask.shape
    lp = policy.logprobs(prompt, resp)
    assert torch.allclose(lp[0, : group[0].response_length], group[0].old_logprobs, atol=1e-4)
