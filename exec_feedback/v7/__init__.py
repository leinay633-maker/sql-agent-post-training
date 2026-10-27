"""v0.7: execution-verified distillation from a 32B teacher vs self-RFT on the 4B student, plus a 14B self-RFT arm.

Reuses the v0.5 task registration and the v0.5 R/P adapters (imported, hash-verified), the v0.5 selection rule and
LoRA trainer unchanged, and the v0.6 vLLM generation path. See docs/RESEARCH_V07.md.
"""

PROTOCOL = "efa-v7-teacher-distill-v1"
MODELS = {
    "student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "teacher": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
    "scale": ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18"),
}
HYBRID = frozenset({"teacher", "scale"})

# Qwen3 hybrid-thinking models in non-thinking mode. The official template drops the empty think block from history
# turns but adds it to the latest one, so a multi-turn dialogue is not prefix-consistent and per-turn loss masks are
# impossible. Here every assistant turn, history and generation prompt alike, carries the same empty think block.
NON_THINKING_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
)
