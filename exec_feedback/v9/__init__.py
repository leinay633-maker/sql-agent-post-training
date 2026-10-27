"""v0.9: online RL with an execution-verified terminal reward, started from v0.8's final family PTR, against online
rejection-sampling fine-tuning (RAFT) on the same kind of online samples.

One run = one GPU: a vLLM engine samples G multi-turn dialogues per pool task with the current LoRA (re-read from
disk after every update), the train-only oracle executes each final SQL against the gold result, and a PEFT trainer
takes one on-policy step on every sampled assistant token. RL weights each trajectory by its leave-one-out advantage
(RLOO); RAFT trains on the correct trajectories only (REINFORCE without a baseline). See docs/RESEARCH_V09.md.
"""

PROTOCOL = "efa-v9-online-rl-v1"
MODELS = {"student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")}
# The v0.8 run whose PTR adapters every RL run starts from (b51e3635..., 2026-10-12 on 8xH20).
V08_PROTOCOL = "efa-v8-gated-teacher-v1"
V08_RUN = "b51e363579307af300a760987abca04840ad815c37ddca2a9c2f0dc2fc67eca7"
START = "PTR"
# family -> advantage estimator; RLhi is RL at the exploratory higher learning rate on two seeds
METHODS = {"RL": "rloo", "RAFT": "raft", "RLhi": "rloo"}
