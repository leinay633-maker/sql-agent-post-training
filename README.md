# SQL 数据分析 Agent 后训练

Post-training Qwen3-4B-Instruct-2507 as a multi-turn SQL analysis agent on BIRD with execution feedback. Rejection-sampling fine-tuning, DPO, teacher distillation and online RL (RLOO) are compared version by version, each under a pre-registered protocol. Docs are in Chinese.

给 Qwen3-4B-Instruct-2507 做多轮 SQL 数据分析 Agent 的后训练。模型在只读 SQLite 沙箱里先看表结构、试跑查询，看到结果再提交 SQL，对错只看执行结果。从 v0.2 到 v0.12，依次比较了拒绝采样微调、DPO 偏好优化、大模型蒸馏和在线强化学习。每版上机前先写预注册协议，定下唯一的主比较和判定标准，跑完照标准判成立或不成立，不成立的也照实记录。

这是开发仓库的公开快照：代码、各版研究协议、实验配置和测试都在；上机操作手册和开发日志没有公开，数据、模型权重和运行产物也不在仓库里。

## 结果

同一晚的评测：BIRD 官方 dev 去掉参考 SQL 预检不通过的 33 题，共 1501 题，贪心解码，三个训练 seed 取平均。

| 系统 | 官方执行准确率（EX） |
|---|---|
| Qwen3-4B-Instruct-2507 原版 | 49.5% |
| 最终系统（离线训练起点 + 边界题在线 RL） | 57.9% |

- 最终系统比原版高 8.4 个点，95% 区间 [+6.2, +10.7]，区间按数据库聚类 bootstrap。
- 最后一步只在离线起点时对时错的边界题上做在线 RL。按比官方 EX 更严的项目判分，它比离线起点再高 2.6 个点，区间 [+0.7, +4.7]，过了预注册检验。
- 把预检排除的 33 题都算作答错，全部 1534 题上是 48.4% → 56.7%。
- 这是同一套设置下的相对提升，不是排行榜成绩：不给列描述，Agent 最多 4 步，单次贪心、不投票。

有正式判定的预注册比较：

| 版本 | 比较 | 结论 |
|---|---|---|
| v0.3 | 修复上下文 SFT 对普通成功答案 SFT | 不成立，反而低 17–20 个点 |
| v0.5 | 执行验证的多轮拒绝采样微调对原版 | 成立，validation +2.0 个点 |
| v0.7 | 老师轨迹补到学生做不对的题上 | 成立，勉强过线 |
| v0.7 | 等量的 32B 老师数据对学生自己的数据 | 不成立 |
| v0.8 | 过了门槛的 235B 老师补题对只用学生数据 | 成立，test +2.4 个点 |
| v0.8 | 带金标提示的自我纠正 | 不成立 |
| v0.9 | LoRA 在线 RL（RLOO）对离线起点 | 不成立：平均 +1.4 个点，下界 −0.1；和只学正确轨迹的 RAFT 也分不开 |
| v0.10 | FSDP2 全参数在线 RL 对 LoRA 在线 RL | 低 4.2 个点，按预注册的反向标准判为更差 |
| v0.11 | 边界题在线 RL 对离线起点 | 成立，+2.6 个点 |
| v0.11 | 边界题在线 RL 对普通在线 RL | 不成立，分不开 |
| v0.12 | 在线蒸馏对原版 | 成立，+4.7 个点 |
| v0.12 | 在线蒸馏对离线蒸馏 | 不成立 |

各版的完整协议在 `docs/RESEARCH*.md`。

## 做法

- **数据**：BIRD 官方训练集按数据库切成训练题池、dev 和 validation，官方 dev 留作最终 test，同一个数据库只出现在一个集合里；参考 SQL 先预检，不通过的题排除。
- **Agent 与沙箱**：模型每步输出一个 JSON 动作（看表结构、试跑查询或提交），最多 4 步。SQL 在独立进程的只读 SQLite 里执行，限制 VM 步数、墙钟时间、返回行数和结果大小，截断的结果不判对。评测时模型看不到金标。
- **判分与统计**：除官方 EX 外，主比较用更严的项目判分（结果按多重集合比较，NULL、浮点和排序单独处理）；区间按数据库聚类 bootstrap。
- **离线训练**：对执行验证过的整条多轮轨迹做拒绝采样微调（LoRA），报错的轮次不计损失，再接 DPO。大模型老师先在 dev 上证明比学生强 5 个点以上，才用来蒸馏、补到学生做不对的题上。
- **在线训练**：以执行结果为奖励做 RLOO，LoRA 每步热更新到 vLLM 采样端。对照有 FSDP2 全参数版本、只学正确轨迹的 RAFT 和在线蒸馏（逐 token 反向 KL）。最后一版只在离线起点时对时错的边界题上训练。

## 目录

| 路径 | 内容 |
|---|---|
| `exec_feedback/` | 数据切分、Agent 交互、沙箱、判分、训练、评测和报告；`rft/`、`rl/` 和 `v6/`–`v12/` 是各版实验的编排，`web/` 是分析服务的页面 |
| `docs/` | `RESEARCH*.md` 是各版预注册协议，`DIAGNOSIS_V10.md` 是 v0.10 全参数 RL 的诊断，`OPERATIONS.md` 是安装和服务说明 |
| `configs/` | 各版实验配置 |
| `scripts/` | 各版无人值守运行、打包和导出脚本；`verify.py` 是本机验收 |
| `tests/` | 单元测试和端到端流程测试 |
| `reports/` | 早期版本的验收报告 |

## 运行

需要 Python 3.11 及以上。Windows：

```powershell
.\setup.ps1 -ML
.\.venv\Scripts\python.exe scripts/verify.py --out outputs/acceptance-local --require-ml
```

Linux 用 `bash setup.sh`，GPU 机加 `--gpu`。常用入口：

```text
python -m exec_feedback check
python -m exec_feedback fetch-bird --help
python -m exec_feedback plan --config local.json
python -m exec_feedback serve --study <study 目录>
```

完整实验需要自己下载 BIRD 和 Qwen 权重，并准备能放下 4B LoRA 训练和 vLLM 采样的 GPU；各版的编排入口在 `exec_feedback/` 下对应的子包里，配置在 `configs/`。

## 说明

- 代码主要由 AI 编程助手编写；研究问题、协议、判定标准和结果核对由作者负责。
- BIRD 数据和 Qwen 模型按各自的许可使用，仓库里不包含它们。
- 代码以 MIT 许可发布，见 `LICENSE`。
