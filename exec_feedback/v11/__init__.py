"""v0.11: v0.9's online RL from PTR, trained only on each start's boundary tasks.

A pre-pass samples every pool task G times with the seed's own start (its PTR adapter, the training protocol and
temperature) and keeps the tasks it solved sometimes but not always: only those give an RLOO group a non-zero advantage
(v0.9 trained on 13-20% informative groups). Everything else is v0.9's RL: one GPU per run, vLLM samples with the current
LoRA, PEFT takes one on-policy step. The arms it is compared with are the PTR starts and v0.9's RL at its registered
milestone, re-evaluated the same night. See docs/RESEARCH_V11.md.
"""

PROTOCOL = "efa-v11-boundary-rl-v1"
MODELS = {"student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")}
# The v0.8 run whose PTR adapters every run starts from (b51e3635..., 2026-10-12 on 8xH20).
V08_PROTOCOL = "efa-v8-gated-teacher-v1"
V08_RUN = "b51e363579307af300a760987abca04840ad815c37ddca2a9c2f0dc2fc67eca7"
# The v0.9 run whose RL milestone adapters are the comparison arm (a21767fe..., frozen 2026-10-19 12:21 on 8xH20).
V09_PROTOCOL = "efa-v9-online-rl-v1"
V09_RUN = "a21767fe09fa422792663999bb73152cd90a1d9e5bf9ac9dad17e7ca82d92d1a"
START = "PTR"
FAMILY = "RLb"   # RL on the start's boundary tasks
ARM = "RL9"      # v0.9's RL (imported as RL_seed<N>) at its registered milestone
