"""v0.12: on-policy distillation (OPD) from the 235B teacher v0.8 distilled offline, against that offline distillation.

The student (the base 4B with a fresh LoRA) samples its own dialogues as v0.9's online runs do. The teacher (v0.8's FP8
235B, tensor-parallel over four GPUs, loaded once and shared by every run) returns the log-probability of each sampled
assistant token given the same context; the per-token signal is teacher minus sampler log-probability (the negative
single-sample reverse KL, discount zero), trained with v0.9's loss. No gold enters the gradient. The arms are v0.8's TR
(offline: execution-verified trajectories, the student's own plus the teacher's on its unsolved tasks), PTR and the base
4B, re-evaluated the same night; the teacher's missing test score is filled in first. See docs/RESEARCH_V12.md.
"""

PROTOCOL = "efa-v12-opd-v1"
MODELS = {
    "student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "teacher": ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "e156cb4efae43fbee1a1ab073f946a1377e6b969"),
}
# The teacher scores the student's token ids as they are, so these files must be byte-identical (checked at the freeze;
# merges.txt differs only in its '#version' header line and generation_config.json only in sampling defaults).
SHARED_TOKENIZER = ("tokenizer.json", "tokenizer_config.json", "vocab.json")
# The v0.8 run whose TR and PTR adapters are the arms (b51e3635..., 2026-10-12 on 8xH20).
V08_PROTOCOL = "efa-v8-gated-teacher-v1"
V08_RUN = "b51e363579307af300a760987abca04840ad815c37ddca2a9c2f0dc2fc67eca7"
FAMILY = "OPD"
ARMS = ("TR", "PTR")
TEACHER = "B235"
# Exploratory, added before any run: SFT then OPD, i.e. OPD continuing v0.8's TR LoRA (TR is the supervised step here),
# one seed, at low priority with the teacher so the main runs keep their throughput.
EXPLORE = "TROPD"
EXPLORE_START = "TR"
