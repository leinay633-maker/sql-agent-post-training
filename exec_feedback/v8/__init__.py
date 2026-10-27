"""v0.8: pilot-gated distillation from Qwen3-235B-A22B-Instruct-2507 (FP8) into the 4B, against gold-hinted
self-rationalization on the same tasks.

v0.7 failed because its 32B teacher was no stronger than the 4B on this protocol, and nobody had measured that before
the run. Here the teacher first has to beat the 4B on dev (the gate), and both new data sources only cover the pool
tasks the 4B never solved in v0.5 round 1. Reuses the v0.5 registration, R/P adapters, selection rule and LoRA
trainer, and the v0.6/v0.7 vLLM path. See docs/RESEARCH_V08.md.
"""

PROTOCOL = "efa-v8-gated-teacher-v1"
# Same chat template and tokenizer files (checked against the pinned revisions), so teacher dialogues train the
# student without any re-rendering.
MODELS = {
    "student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "teacher": ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "e156cb4efae43fbee1a1ab073f946a1377e6b969"),
}
# Added to the public user JSON only while the 4B samples hinted attempts (HR); build removes both keys again, so
# training prompts are byte-identical to the unhinted v0.5 prompt.
HINT_KEYS = ("reference_sql", "reference_note")
# The 4B queries before finishing on every task; the note keeps that habit (v0.7's T inherited a teacher's habit of
# finishing without a query on about a third of the tasks).
HINT_NOTE = ("Private hint for this practice attempt: reference_sql is known to return the correct result. Solve the "
             "task as usual: check the data with a query first, then finish with SQL that returns the same result.")
