# 接入、部署与故障处理

## 三种部署

**本机 Windows + 云端 GPU：** Windows 用 `setup.ps1` 安装控制/服务依赖；在 Linux GPU 主机克隆同一项目，`bash setup.sh --gpu`，接入数据和固定模型。训练用云端 `study --execute`。浏览器通过 SSH 隧道连接 GPU 主机 8787，不必把模型权重搬回本地。两个机器的路径分别配置，不直接使用对方的绝对路径。

**全部在 Linux GPU：** 数据、训练、服务使用一份 local.json。用 `serve --study` 自动选择验收过的 repair adapter；`--variant success` 查看对照。模型逐阶段卸载，一次只有一个阶段占用模型资源。

**已有兼容 API：** local.json 设置 HTTP 后端、准确的 served model ID 和 endpoint；`EFA_API_KEY` 只在环境中设置。可做 `baseline`、分析页面，不可把闭源 API 当成 QLoRA 训练基座。完整训练仍需要可加载的模型权重。

## 依赖与配置

Python 3.11+，推荐 Linux GPU 用独立虚拟环境。安装后保存 `python -m pip freeze > outputs/installed-packages.txt` 和 `check --probe-gpu` 输出。已固定直接依赖版本；安装解析及 CUDA 二进制兼容性以目标环境实测为准。

初始配置是单卡预算起点，不是性能承诺。先降低 `training.max_steps` 到 20，并将 `experiment.min_pairs`、样本量和调用上限设为一致的试跑范围。成功通过 adapter 重载后再用新 study 名称放大实验。长 schema 不能偷偷截断；按拒绝报告明确调整上下文或数据范围。

模型权重下载需 `allow_download=true`，或提前下载到 local_path。联网下载来自明确记录的模型仓库及固定 revision；不要把来源不明的 pickle/模型文件当作可运行输入。第三方数据库和权重默认被 Git 忽略。

## 默认限制

配置文件严格拒绝未知键。每题交互和输出 token 有上限，SQLite 另有秒数、VM 指令数、行数和字节限制；超限记失败，不拿截断答案判正确。HTTP 没有自动重试，不将模型 ID 不匹配的结果混进基线。

持久化 CallBudget 在请求前预留一次调用；被杀掉的半次请求可能保守计数，不能为了凑额度抹去。study、服务分别使用自身输出目录下的预算文件。服务器预算用完返回错误/429；不要删除记录继续声称同一个预算实验，应另开有明确额度的目录。

软件退出、GPU 内存释放、云实例停止计费是三件事。使用云主机时，应另外设置平台自动释放或手动停止，并检查磁盘计费。项目不会未经授权操作云账户。

## 输出与恢复

`outputs/studies/<name>/study.json`：状态、固定配置、阶段证据哈希。

`base_*/eval_*/collect/`：模型/数据/协议清单、逐题轨迹、汇总、完成哈希。

`pairs/`：两组配对 SFT、真实/规则来源、被拒样本、选中轨迹。自然轨迹不够时，natural 模式停止；mixed 才会用明确标记的规则增强。

`train_*/`：训练清单、checkpoints、真实 loss 日志、adapter、保存重载校验。完整 checkpoint 含文件哈希；损坏即拒绝加载。

`report_*/`：独立 HTML、Markdown、CSV、JSON。不同 seed 各自报告，按数据库聚类估计配对差异区间。

重跑同一 study 命令自动跳过校验通过的阶段。代码/配置/数据改变须新名称；已完成结果变更会报错，不能通过覆盖结果掩盖。未完成衍生语料目录会留存并从已有采集轨迹重新构建。ZIP 下载/解压中断与训练恢复不同，下载器会明确报告而不是伪装完整。

## 常见错误

| 错误 | 处理 |
|---|---|
| CUDA unavailable | 在实际 GPU 主机装匹配驱动/torch，明确一张可见 GPU；不在 Windows 控制端强行训练 |
| 模型缓存不存在 | 接入正确 local_path，或明确允许下载固定版本 |
| Reference-query preflight failed | 查看 gold.jsonl 中真实错误；核对数据版本/路径/限制，不删除坏题掩盖问题 |
| Not enough length-checked pairs | 查看筛选原因；扩大训练候选或调整模式/长度，使用新 study 名称 |
| Context budget exceeded / OOM | 核对 schema 和轨迹长度，降低单批/长度或换显存；记录变更，不宣称之前跑过 |
| Study config/code changed | 新建 name，保留旧结果 |
| Adapter reload verification failed | 不能对外当作完成模型；核对基座、精度与依赖，保留 result.json |
| HTTP model id mismatch | 请求实际 served name，不通过关闭检查混用其他模型 |
| 公网绑定失败 | 设置强 EFA_SERVER_KEY，配置 TLS 或 SSH 隧道；默认只绑定 localhost |

## 数据与安全边界

BIRD 入口：<https://bird-bench.github.io/>。遵守原始数据许可，不把业务数据或其他内部数据用于公开训练。源 JSON 的 SHA-256 证明本次输入固定，不证明来源可信或全部题目正确。

服务当前是单用户研究工作站，不是面向不可信客户的多租户平台。数据由受控路径发现，模型无任意文件工具。SQLite 防护仍不是 OS 级安全证明；处理不可信数据库，应使用非 root、只读挂载、无外网的独立环境。
