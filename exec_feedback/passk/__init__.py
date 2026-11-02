"""pass@k capability boundary: does RL only re-rank answers the model it started from could already sample?

No training. Ten frozen models (the base 4B B; v0.8's PTR, the RL start; v0.11's RLb at its registered milestone; v0.12's
OPD at its registered milestone; three seeds each for the trained families) each sample every test task `samples` times
with the agent protocol, budget and temperature the online runs trained with, graded by the project's grader against the
frozen gold cache. pass@k (unbiased estimator) is compared at k = 1 .. samples with v0.6's database x seed cluster
bootstrap; tasks whose gold result is a single cell are reported apart (a wrong query can hit one value by chance).
See docs/RESEARCH_PASSK.md.
"""

PROTOCOL = "efa-passk-v1"
MODELS = {"student": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")}
# The v0.8 run that trained PTR (b51e3635..., 2026-10-12 on 8xH20); v0.11 started RLb from these adapters.
V08_PROTOCOL = "efa-v8-gated-teacher-v1"
V08_RUN = "b51e363579307af300a760987abca04840ad815c37ddca2a9c2f0dc2fc67eca7"
# v0.11 (7b9bca19..., 2026-10-26 on 1xH20): RL from PTR on each start's boundary tasks, registered milestone K = 200.
V11_PROTOCOL = "efa-v11-boundary-rl-v1"
V11_RUN = "7b9bca1979445eed31f5ed04dfa105f8fd55dbc56a36710d643bb0c417483247"
# v0.12 (8834421c..., 2026-10-26 on 8xH20): on-policy distillation from the base 4B, registered milestone K = 260.
V12_PROTOCOL = "efa-v12-opd-v1"
V12_RUN = "8834421c0175d161b0ca8b21ff50fce6805ff6ec0647d06429c351772cc0b518"
BASE, START, RL, OPD = "B", "PTR", "RLb", "OPD"
FAMILIES = (BASE, START, RL, OPD)
# the registered comparison points the study samples RLb and OPD at (the freeze refuses any other step)
MILESTONES = {RL: 200, OPD: 260}
