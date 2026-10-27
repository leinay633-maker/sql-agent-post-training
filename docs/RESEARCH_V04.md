# v0.4：终局执行奖励的在线多轮 RLOO

状态：方法与工程实现；没有 H20/BIRD 模型结果。CPU 测试不是训练效果。本协议须在 v0.3 正式 study 完成、结果包拉回本机且用户同意后才合并；下一 GPU 时段长度尚未确定。

## 1. 研究问题与可识别的比较

问题：从一个固定的执行反馈 SFT adapter 出发，在线采样真实工具轨迹，用最终 SQL 执行匹配奖励优化全部助手动作，能否改善相同预算下的 SQL Agent 成功率？

| 原设计 | 新设计 | 原因与限制 |
|---|---|---|
| v0.3 `efa-v3`，A/B/C/D，主比较 D−C | 新 `efa-v4-rloo-two-turn-v1`；S4 冻结 repair SFT，E4 在线 RLOO；主比较 E4−S4 | 新任务集合、推理预算与方法必须重新评测，不能拿旧 D 分数充当 S4 |
| 只监督最后 finish；历史错误动作和工具消息均 mask | 终局奖励作用于两轮中全部实际采样的助手 token；工具/问题/schema 仅作条件 | 学的是带执行反馈的轨迹策略，不只是再次拟合最终 SQL |
| 最多 4 次助手生成，512 输出 token | 最多 2 次助手生成，每次 384 token，真实结果复核仍启用 | 明确牺牲长程探索空间以控制在线采样成本；只称两轮工具策略，不称任意长程 Agent |
| 固定 generate batch 4 | 活跃对话补位、按 prompt 长度分桶、batch 16；评测最多 64 对话槽位 | 减少 padding 与已结束对话占位；不是 token 级 continuous batching，也不是 vLLM |
| 512 个 SFT 候选、256 验证题、单 seed | 1024 个 RL 训练题、512 验证题、RL seeds 42/43/44 | 从原 prepared 数据沿用数据库划分，按元数据平衡抽样；实际是否有足够题目由准备阶段核验 |
| v0.3 的两套 SFT 独立比较 | 三个 E4 都从预先指定的 `train_repair_42/adapter` 出发 | 这是条件于同一 SFT 热启动的 RL seed 方差，不是三个 SFT seed 的复现 |

不改 v0.3 的 agent/corpus/train/optimization/study/evaluation/models、配置或 `efa-v3` 字符串。新入口是 `python -m exec_feedback.rl`，旧 CLI 不导入它。旧包元数据和依赖保持原样；研究版本以协议、配置和运行指纹区分。

整个源代码指纹会因新增 Python 文件而变化，这是正常的代码身份变化：不能拿合并后的目录继续旧 v0.3 study。已有正式 study 继续使用 `main@5ea1515` 的原源码包；不能用“老协议没改”绕过其代码指纹检查。

## 2. 为什么选 RLOO，而不是堆叠一个 RL 框架

采用 REINFORCE leave-one-out 思路 [1]，并非声称复刻论文或 TRL 的全部训练细节。每个 batch 用当前策略为 B 个题各采 G 条两轮轨迹，随后只进行一次优化器更新，不复用旧轨迹。

对同题第 i 条轨迹，令终局奖励 `r_i ∈ {0,1}`：

```text
A_i = r_i - sum(r_j for j != i) / (G - 1)
L = - 1/(B*G) * sum_i A_i * sum_{assistant actions, sampled tokens} log p_theta(token | actual prefix)
```

优势不反传，按轨迹数归一化，不按输出长度取平均，不用组内标准差缩放。有限轨迹的所有助手动作共享终局优势，这是粗粒度终局信用分配，不是已学到逐步奖励。组内其他样本的奖励构成不依赖当前样本动作的 baseline；在理想同策略采样条件下，这是标准 score-function 估计。有限精度、裁剪梯度和 AdamW 不等于数学意义的无误差实现。

每批更新前策略未改变，所以不需要 PPO 的旧策略比率，也不能随后私自增加多 epoch。若要复用 rollout 或加入外部推理引擎，必须另外设计行为策略 logprob、比率/裁剪与权重同步。本实现不以“GRPO”命名，避免把不同目标混为一谈。

相较 PPO，暂不增加 critic/value loss；相较带多轮复用和 KL 的 GRPO，少一组对齐与超参数变量；相较 DPO/只筛成功的 RFT，保留当前策略在线探索、成功与失败样本的相对优势，并训练工具动作。RFT/继续 SFT 是有价值的额外对照，但本轮没有实现，不能宣称 RLOO 优于它们。

**KL 系数固定 0。** 没有 reference 模型、奖励模型或人工格式加分；这降低额外前向与显存开销，也失去对策略偏离的显式约束。学习率 1e-5、梯度裁剪 1.0、固定 128 批更新只是预注册设计，不是已验证稳定的最优参数。出现退化、长度膨胀或全错组时如实报告；只能用 train pilot 做工程/预算判断，不能按验证分数挑学习率或 checkpoint。

## 3. 奖励、输入隔离与失败口径

`TrainingOracle` 只接受审计后的 train 任务，研究模式还要求 source 为 `bird-train` 且不是 fixture。预先独立执行 train gold；无效参考 SQL 使准备/训练失败，不删除题目继续跑。模型只收到问题、evidence、schema、自己生成的动作与真实 SQL 工具反馈。

oracle 的 gold 结果缓存在独立对象里，不进入 policy、prompt、工具返回或 tokenizer。最后提交必须为 `submitted`，没有输出截断/生成失败；候选 SQL 重新经原只读 SQLite 执行器执行，与缓存参考结果用原判分器比较。外部 `correct: true` 和已提供结果不能代替重执行。结果匹配给 1，其余 0；没有基于 gold 的修复提示、重试或中途判对错。

SQL 错误、超时、只读拒绝、结果超限、协议错误、未 finish、上下文拒绝、输出 token 截断均留在预注册分母。无可用采样 token 的失败轨迹梯度为零，但仍占 B×G 的分母。OOM 是致命错误，不伪装为一批正常失败后继续、不自动缩 batch。

每组全 0 或全 1 时优势为 0，不按“有信号”为标准重采样或重加权。整个 batch 无信号时不执行 Adam step，防止动量/权重衰减造成无奖励更新；仍消耗该批计划与题目游标。`optimizer_updates` 和批次 `update` 分开计数。完成 128 批而实际 optimizer_updates=0，不得包装成“RL 学到了策略”。

SQL 执行相等仅是固定数据库快照上的可验证奖励，不能证明查询对所有数据库实例语义等价，也不能排除结果碰巧相同、空集等伪解。原判分器的行数、近似匹配、排序与资源边界继续生效。没有加入 test-suite mutation 等额外语义验证；不声称抗 reward hacking 已解决。

## 4. 采样、训练与精度

训练 `temperature=1, top_p=1, top_k=0, repetition_penalty=1`，单 beam，不继承模型自带 generation_config 的截断采样默认值。因为目标重算的是普通 softmax logprob，行为采样必须与其一致。GenerationConfig 参数定义来自固定 Transformers 文档 [2]。

生成保留每轮真正的 prompt token IDs、completion token IDs（包含实际采到的 EOS）及 finish_reason。损失不用“decode 后再 tokenize”的近似边界。每轮独立 teacher forcing，只对该轮采样 token 计算 logprob；前一轮助手动作会在其自己的 loss 项里训练，不误把所有历史动作永久 mask。历史和工具 token 是上下文，不作为目标。

所有 Dropout 在 RL loop 中禁用，避免采样 eval 与 teacher forcing train 的分布不同。采样使用当前可训练 PEFT adapter，不卸载/重新同步权重；BF16 基座、LoRA 参数、单样本梯度累积沿用现有固定栈。这里只限制 lm_head 输出到动作尾部，并不声称已经实现 FlashAttention、融合 loss 或批量反向的性能。

左 padding 的批量生成与不 padding 的重算在 BF16/SDPA 下可能有数值差异。CPU 用 tiny tensor 和可选 tiny Qwen3+PEFT 检查梯度/保存重载；H20 必须再检查实际 Qwen3-4B、长 schema、显存和有限精度。CPU 通过不能替代它们。

## 5. 吞吐选择与可核查实验规模

暂不引入 vLLM。理由是本轮需要同一个活跃 LoRA 在每个 RL batch 后立即采样；双引擎还需要处理权重版本、KV cache 失效与采样 logprob 一致性，且会引入新的二进制依赖组合。无 GPU 无法证实该组合稳定，不能把“安装了 vLLM”当成吞吐成果。

本轮实现可测量的 batch 16 + 长度分桶 + 对话补位，复用相同 HF/PEFT 依赖；上卡用相同 train 题、同 adapter、同两轮/384 token 预算对比 batch 4 FIFO，记录墙钟、generate 时间、实际 tokens、padding、峰值显存、失败及 SQL 差异。较快但大量提前失败不构成有效加速。训练反向逐动作累积，可能仍是主要瓶颈；比较没有承诺必然更快。

正式配置每 seed：128 批 × 8 题 = 1024 个 prompt draws；固定打乱的一轮恰好覆盖 1024 个训练题。每题 4 条采样轨迹，即每 seed 4096 条、最多 8192 次生成。三个 seed 的训练最多 24576 次；S4 与三个 E4 各 512 验证题 × 2 步，最多 4096 次评测生成。合计上界 **28672 次生成、11010048 个输出 token**，均为配置算术，不是实测数据，不含独立 pilot、重载、gold SQL 或中断后的重放成本。

## 6. 冻结、统计与可复查证据

准备阶段验证 v0.3 study complete、D42 adapter 完成且重载通过、模型/revision、语料哈希、train 来源、数据库字节及任务一致性。新 manifest 冻结任务文件哈希、数据库集合、adapter、源 study/语料、配置和代码。不够 1024/512 时拒绝静默缩小样本；不重划数据库以凑数。

三个 seed 预先登记并全部完成、adapter 哈希/重载检查通过，才开放任何 validation gold 预检或评分。pilot 注册 64 train 题、0 validation，不能走验证入口。固定最终批次，不用验证早停，不选 best seed/checkpoint，不载入保留测试。旧验证结果即使可见，也不是选择 D42、学习率或本协议规模的依据。

报告逐 seed 配对 E4−S4、提升/退化题 ID、错误类型、seed 间标准差；按数据库簇和 RL seed 分别重采样，所有模型始终使用相同抽到的题索引。不能把 3×512 当作 1536 个独立测试题。数据库簇少、只有三个 RL seed 的区间可能很不稳定；只对当前平衡抽样和一个固定 SFT 热启动成立。不是 BIRD 官方评测，也不是最终保留测试结果。

没有完成的条件按预注册分母记 not_attempted，缺少完成封条时不输出正式置信区间。只接受本 run 的协议、题 ID、推理预算、adapter 和封条哈希，拒绝 v0.3 或其他 run 混入。原始 rollouts/metrics/attempts 追加保留；中断后从最近完整 optimizer+LoRA+游标+CPU/CUDA/Python RNG checkpoint 继续，重放的孤立尝试不进入 durable history 的完成批次数。恢复仅承诺同路径、相同软件/硬件配置下的状态恢复，GPU 跨批数值逐位复现未验证。

## 7. 能说与暂时不能说

完成本 PR 后能说实现并 CPU 测试了“终局执行奖励、两轮工具交互、全部助手动作 RLOO、分桶推理与可审计的多 seed 协议”。不能说已经提升成功率、降低成本、在 H20 收敛、达到某吞吐或具备统计显著性。

只有上卡完成后，才能用真实样本数、失败率、区间、token/时间成本和代表性回归样例报告结果。E4−S4 的正结果也只说明“额外 RL 阶段相对固定 SFT 起点的变化”；没有计算匹配的继续 SFT/RFT 对照，不能归因为“RL 算法一定更优”。三 seed 共用一个 warm start、两轮限制、无 KL 和 reward hacking 是必须主动说明的边界。

## 参考与实现范围

[1] Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (2024): https://arxiv.org/abs/2402.14740 。本项目借鉴 leave-one-out 的 baseline 思路，奖励改为 SQLite 执行匹配，不引用论文的速度/效果作为本项目结果。

[2] Transformers 4.57.3, GenerationConfig: https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/text_generation 。只用于参数语义与 API 核对；目标环境兼容性仍待 H20 验证。

实现地图：`rl/objective.py` 数学目标；`backend.py` 活策略批采样；`rollout.py` 工具轨迹/独立奖励；`training.py` 在线优化；`checkpoint.py` 训练状态；`study.py` 冻结来源与划分；`evaluation.py` 评测门禁和 train 吞吐对照；`reporting.py` 配对统计；`config.py` 配置审计；`__main__.py` 显式阶段入口。没有新增云服务、租期管理、打包器、前端或付费 API。
