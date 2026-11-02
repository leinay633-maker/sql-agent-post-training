"""Deployment study: one trained model served on one H20 three ways, and what each way costs.

The model is v0.11's RLb_seed44 at its registered milestone (the best of the three RLb seeds on dev). The three
deployments: the base 4B with the LoRA (`lora`), the LoRA merged into bf16 weights (`merged`), and the merged weights
quantized to FP8 as vLLM loads them (`fp8`). Greedy evaluations on test and validation measure what merging and FP8
cost; a load test drives the analysis service (whole agent dialogues) against vLLM's OpenAI-compatible server at
several concurrency levels; an alignment check compares the serving path with the evaluation path on the same tasks.
Descriptive: no registered decision. See docs/RESEARCH_DEPLOY.md.
"""

PROTOCOL = "efa-deploy-v1"
MODELS = {"student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")}
# v0.11 (7b9bca19..., 2026-10-26 on 1xH20): its RLb runs at the registered milestone K = 200.
V11_PROTOCOL = "efa-v11-boundary-rl-v1"
V11_RUN = "7b9bca1979445eed31f5ed04dfa105f8fd55dbc56a36710d643bb0c417483247"
FAMILY, SEED = "RLb", 44
MODEL = f"{FAMILY}_seed{SEED}"
CONDITIONS = ("lora", "merged", "fp8")
SERVED = "rlb44"  # the model name the vLLM server answers to (the LoRA module's name in the lora deployment)
