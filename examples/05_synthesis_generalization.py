"""Leave-one-molecule-out evaluation of the tabular synthesis policy.

Held-out numbers score dataset records with the rule reward (policy-independent);
training rewards track the fitted value/actor networks on the training molecules.
"""

from forgeline.domains.synthesis import TabularActorCritic, TabularPPOConfig, TabularPPOTrainer, load_trajectories, rule_score
from forgeline.evaluation import RecordRewardSuite

rows = load_trajectories("data/samples/synthesis/trajectories_literature.jsonl")
for held_out in sorted({r["molecule"] for r in rows}):
    train = [r for r in rows if r["molecule"] != held_out]
    test = [r for r in rows if r["molecule"] == held_out]
    metrics = TabularPPOTrainer(TabularActorCritic(), TabularPPOConfig(epochs=2)).train(train)
    heldout = RecordRewardSuite(test, rule_score).run().metrics
    print(f"{held_out:12s} train_reward={metrics['final_reward']:.3f} heldout_record_reward={heldout['mean']:.3f}")
