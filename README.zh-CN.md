# Raven Trainer

[English](./README.md) | **简体中文**

工业视觉**数据资产与模型训练平台**：把原始视频变成可版本化的数据集，按 Recipe 物化出可复现的
训练工作区，通过可插拔后端完成训练，并导出可直接部署的 ONNX + 元数据。

Raven Trainer 是 **Raven-Go**（运行时/引擎）的训练侧配套系统，包含 FastAPI 后端 + React 控制台、
Click 命令行（`raven-train`）以及一组基于 `uv` workspace 的后端适配器。

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.11-3776AB" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue" />
  <img alt="Tests" src="https://img.shields.io/badge/tests-75%20passed-brightgreen" />
  <img alt="mypy" src="https://img.shields.io/badge/mypy-0%20errors-brightgreen" />
</p>

---

## 设计出发点

数据集是「代码」，不是一堆文件。**Dataset Recipe** 是唯一事实来源：用它声明式地描述数据集
*如何*被构建（来源、采样、质量门槛、标注、切分）。维护 Recipe，重新生成数据——同一份 Recipe
永远产出同一个数据集，每个产物都可追溯。

## 三层存储模型

```
storage/      原始层（不可变）    导入的视频/图片，永不原地修改
   ↓
datasets/     资产层（版本化）    <dataset>/versions/vN/{manifest.jsonl, snapshot.yaml, annotations/}
   ↓
workspace/    训练层（物化产物）  train/ val/ test/ + data.yaml，可直接交给训练器
```

`recipes/` 独立于三层之外：它保存连接 `datasets/` 与 `workspace/` 的构建规则。

版本是**只追加**的：版本目录原子出现，显式 tag 冲突直接报错，任何模块都不允许手写 manifest——
所有写入统一走 `VersionManager`。

## 流水线

```
原始视频 → 采集（采样 / 质量门槛 / 去重） → 注册 → 版本
        → Recipe → 工作区（切分 + data.yaml） → 训练 → 导出（ONNX + 元数据）
```

| 阶段 | 模块 | 职责 |
|---|---|---|
| 采集 | `raven_trainer/ingestion` | 抽帧采样（interval / motion / burst / object-driven / scene-change）、模糊 / 全黑 / 过曝 / 小目标剔除、dHash + SSIM 去重 |
| 注册与版本 | `raven_trainer/data_asset` | 注册表、manifest、原子版本快照、标签、血缘、SQLite 元数据、反馈合并 |
| Recipe → 工作区 | `data_asset/recipe`、`data_asset/workspace` | 声明式构建配方、确定性 train/val/test 切分、各后端的数据布局 |
| 训练 | `raven_trainer/adapters`、`experiments/` | 后端适配器（Ultralytics / MMDetection / Anomalib / RF-DETR）、实验记录、日志实时推送 |
| 导出 | `raven_trainer/exporters`、`scripts/export_obb_onnx.py` | 按 family 约定（NMS、动态尺寸）导出 ONNX，并生成元数据 sidecar |

标注统一以 **CVAT XML / COCO JSON** 存储（无损、可被标注工具直接读写），不使用 YOLO txt；
YOLO / COCO 目录结构在构建工作区时才生成。

## 快速开始

需要 Python ≥ 3.11 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/DavidSche/raven-trainer.git
cd raven-trainer
uv sync                      # 仅核心（不装 torch）
uv sync --extra ultralytics  # 装一个后端：ultralytics | rfdetr | anomalib | mmdetection
uv sync --extra all          # 全部后端

# 1. 新建业务项目（只有配置，没有代码）
uv run raven-train new --name person_intrusion --task detection --model yolo26n

# 2. 原始视频 → 版本化数据集
uv run raven-train ingest run --help          # 采样 / 质量 / 去重 / 场景分析参数
uv run raven-train dataset create person_v1
uv run raven-train dataset version person_v1 --create --source data/ingestion/runs/<run>/kept

# 3. 按 Recipe 构建训练工作区
uv run raven-train dataset recipe person_recipe --create
uv run raven-train dataset recipe person_recipe --build

# 4. 训练与导出
uv run raven-train train projects/person_intrusion/train.yaml --epochs 100
uv run raven-train export projects/person_intrusion/project.yaml --register

# 5. Web 控制台（FastAPI + React，http://127.0.0.1:8081）
uv run raven-train ui --open
```

所有状态都落在仓库根目录的 `data/` 下，可用 `RAVEN_DATA=/path` 覆盖。

### 命令一览

| 命令 | 作用 |
|---|---|
| `raven-train new` | 生成业务项目骨架（`project.yaml`、`train.yaml`、`dataset.yaml`） |
| `raven-train ingest` | 采集流水线：`scan` · `run` · `profile` · `report` |
| `raven-train dataset` | 数据资产：`create` `list` `show` `version` `recipe` `build` `annotate` `task` `doctor` `lineage` `stats` `analyze` `tag` `misdata` |
| `raven-train train` | 通过后端适配器训练，并写入实验记录 |
| `raven-train eval` | 在数据集版本上评估已训练模型 |
| `raven-train export` | 导出 ONNX（含元数据），供 Raven-Go / raven-onnxruntime 使用 |
| `raven-train job` | 查看统一任务中心：`list` · `show` · `log` |
| `raven-train ui` | 启动 FastAPI 后端与 React 控制台 |

### HTTP API

后端挂载在 `/api/v1`（datasets、versions、annotations、ingestion profiles、training jobs、
models、pipeline、feedback、reports、tasks、storage、system）以及 `/api/health`；服务运行时可在
`/docs` 查看 OpenAPI 交互文档。

采集、训练、导出、流水线等长任务统一进入任务中心，并通过 WebSocket 推送进度；涉及磁盘或 CPU 的
接口都是同步端点，由线程池执行，不会阻塞事件循环。

## 后端适配器

后端是 `uv` workspace 中的独立分发包，因此核心安装保持轻量（不带 torch），重依赖由各适配器
自行声明。

| 包名 | 后端 | 任务 |
|---|---|---|
| `raven-ultralytics` | Ultralytics（YOLO26 / YOLO11） | 检测、分割、姿态、OBB、分类 |
| `raven-mmdetection` | MMDetection（RT-DETR） | 检测 |
| `raven-anomalib` | Anomalib | 异常检测 |
| `raven-rfdetr` | RF-DETR | 检测、分割 |

新增后端只需实现 `raven_trainer/adapters/base.py` 的适配器契约，训练器、导出器与 UI 无需改动。

## 目录结构

```
raven_trainer/
├── config.py        # R：路径助手 + validate_name（唯一的名称/路径闸门）
├── data_asset/      # 注册表、版本、recipe、workspace、元数据、标签、血缘、
│                    #   标注任务、反馈、任务中心、体检、脏数据、切分
├── ingestion/       # profile、pipeline、采样、质量、去重、场景、报告
├── datasets/        # 数据集配置 + 转换器（CVAT / Label Studio）、统计
├── adapters/        # 适配器契约 + 内置后端
├── experiments/     # 实验记录（exp_NNNN）与产物
├── exporters/       # ONNX 导出辅助
├── api/             # FastAPI 应用与路由
└── cli/             # raven-train 命令树
adapters/            # workspace 包：raven-{ultralytics,mmdetection,anomalib,rfdetr}
ui_src/              # React 19 + Vite + TypeScript 控制台
scripts/ · tools/    # 运维脚本（导入、导出、数据集检查）
tests/               # pytest 用例（安全、API 冒烟、CLI）
```

## 开发

```bash
uv run pytest -q                     # 75 个用例
uv run mypy raven_trainer            # 0 error
uv run pylint raven_trainer          # 8.77 / 10
python -m compileall -q raven_trainer
```

`tests/` 故意没有 `__init__.py`，因此检查测试文件时需显式列出：
`uv run pylint tests/test_safety.py tests/test_api_smoke.py`。

开发约定（详见 `AGENTS.md`）：不得修改 `data/`、`.venv/`、`__pycache__/`、`*.egg-info/`、
`dist/`、`build/` 与模型权重；所有用户输入的名称必须经过 `validate_name()`；所有客户端提供的
路径必须解析在 `PROJECT_ROOT` 或 `DATA` 之内。

## 文档

数据集、模型、训练、增强、导出、编码规范等详细规格文档目前与内部设计文档一起维护，
**暂未在本仓库公开**。当前仓库内最实用的参考是：`AGENTS.md`、`raven_trainer/config.py`
（路径契约）、`raven_trainer/adapters/base.py`（后端契约）以及服务运行时的 `/docs`。

## 许可证

本项目采用 **Apache License 2.0** 授权，详见 [LICENSE](./LICENSE)。
除遵守许可证条款外，不得使用本项目。
