"""v0.6: iterative execution-verified RFT on a 7B code model, vLLM rollouts, BIRD column notes.

Reuses the v0.5 task registration (same pool / dev / validation / test files, verified by hash), the
v0.5 selection rule and LoRA trainer unchanged, and the gold-free agent/sandbox. See docs/RESEARCH_V06.md.
"""

PROTOCOL = "efa-v6-iterative-rft-v1"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
