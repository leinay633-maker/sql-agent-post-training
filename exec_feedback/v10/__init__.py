"""v0.10: the v0.9 online RL objective with full-parameter training, on a distributed rollout/training system.

One run takes one 8-GPU host: rollout workers (one vLLM engine per GPU) sample whole groups of multi-turn SQL dialogues
with the current full weights; an FSDP2 trainer (fully sharded fp32 master weights, bf16 compute) takes one RLOO step
over every sampled assistant token; after every step the trainer gathers the new bf16 weights and every engine loads
them in place, verified by a weight checksum. RLF samples step s with the weights after step s (on-policy); RLFa samples
it with the weights after step s - 1 while the trainer is still on step s - 1 (one-step staleness, corrected by the same
truncated importance weights). The LoRA arm is v0.9's RL at its registered milestone. See docs/RESEARCH_V10.md.
"""

PROTOCOL = "efa-v10-dist-full-rl-v1"
MODELS = {"student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")}
# The v0.8 run whose PTR adapters every run starts from, merged into full weights (b51e3635..., 2026-10-12).
V08_PROTOCOL = "efa-v8-gated-teacher-v1"
V08_RUN = "b51e363579307af300a760987abca04840ad815c37ddca2a9c2f0dc2fc67eca7"
# The v0.9 run whose RL (LoRA RLOO) milestone adapters are the LoRA arm (a21767fe..., frozen 2026-10-19 12:21 on 8xH20).
V09_PROTOCOL = "efa-v9-online-rl-v1"
V09_RUN = "a21767fe09fa422792663999bb73152cd90a1d9e5bf9ac9dad17e7ca82d92d1a"
START = "PTR"
LORA = "RL"
# family -> rollout staleness in optimizer steps
FAMILIES = {"RLF": 0, "RLFa": 1}
