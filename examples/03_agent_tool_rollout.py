"""Multi-turn tool-use episode with a scripted policy: shows tool execution, result injection and verifiable scoring."""

import torch

from forgeline.core.protocols import Trajectory
from forgeline.data import CharTokenizer
from forgeline.rollouts.agent import AgentRolloutConfig, AgentRolloutEngine, build_agent_prompt
from forgeline.rollouts.rewards import build_reward_provider

tok = CharTokenizer.ascii()


class ScriptedPolicy:
    """Stands in for a trained model: first calls the tool, then answers."""

    device = torch.device("cpu")
    pad_token_id = 0
    turns = ['<think>multiply</think><tool_call>{"name": "python_executor", "args": {"code": "print(37 * 43)"}}</tool_call>',
             "<final_answer>1591</final_answer>"]

    def __init__(self):
        self.i = 0

    def encode(self, text):
        return tok.encode(text)

    def decode(self, ids):
        return self.turns[self.i - 1]

    def generate(self, ids, settings):
        self.i += 1
        return torch.zeros(1, 1, dtype=torch.long)


prompt = build_agent_prompt("Compute 37 * 43.")
segments, tool_results, output = AgentRolloutEngine(ScriptedPolicy(), AgentRolloutConfig(max_context_tokens=10_000)).run_episode(prompt)
print("segments:", segments)
print("tool results:", tool_results)
traj = Trajectory(prompt_ids=torch.tensor(tok.encode("x")), response_ids=torch.tensor(tok.encode(output)), response_text=output, task={"answer": "1591"})
print("reward:", build_reward_provider({"type": "tagged"}).score(traj))
