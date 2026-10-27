# 执行入口

项目的可运行流程已在 `README.md` 中给出；本文件不再是一份“待实现核心模块”的清单。

本机操作顺序：安装隔离依赖 → `init` → 接入数据/模型 → `check` → `prepare`/`validate-gold` → `plan` → `study --execute` → `serve --study ...`。

接入验收：`python scripts/verify.py --out outputs/acceptance-local --require-ml`。需要验证 GPU 的真实 4B 路径时，先用新 study 名称、低训练步数运行完整流程，再扩大样本。修改配置不得混入旧 study。

真实数据、GPU 及模型资源由操作者提供；本次交付中尚未发生的真实实验由命令执行后产出，不用演示或配置文件替代。
