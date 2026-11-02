# raven-trainer (Raven Trainer)

Raven AI 工业视觉数据资产与模型训练平台，raven-go 的训练侧配套：核心是**数据管线**（约 60% 代码）+ 模型训练/导出，产物为带元数据的 ONNX。技术栈：Python ≥ 3.11 + uv（`uv.lock`）+ Click CLI + FastAPI/uvicorn + Pydantic 2 + Math/loguru + 训练适配器位于 `adapters/`（`raven-ultralytics` / `raven-mmdetection` / `raven-anomalib` / `raven-rfdetr`）；`ui_src/` 为 React 19 + Vite + TypeScript 前端。

## Commands
- Install: `uv sync`（Python ≥ 3.11；或 `uv pip install -e .`）；`ui_src/` 单独 `npm install`
- 启动 API + UI：`raven-train ui`（FastAPI 服务，CORS 放行 `localhost:5173`）
- CLI 帮助：`uv run raven-train --help`
- Build / Package: `uv build`
- Lint + Typecheck: `uv run pylint`（配置在 `pyproject.toml`）+ `uv run mypy`
- Test: `uv run pytest`（`testpaths = ["tests"]`，识别 `test_*.py` / `*_test.py`）
- 单测某个文件: `uv run pytest tests/xxx/test_y.py`
- UI（前端子项目）: 进入 `ui_src/` 后 `npm run dev` / `npm run build`（`tsc -b && vite build`）
- OBB 导出（必须走包装脚本，见 EXPORT_SPEC §6）: `python scripts/export_obb_onnx.py --weights yolo26m-obb.pt`，校验 `--verify-only ../path/model.onnx`

## Architecture
- 数据集管线（核心，约 60% 代码）：`raven_trainer/dataset_pipeline/`（importers / cleaners / annotators / versioning / builders / recipes / workspace / manifest）+ `raven_trainer/ingestion/`（import、dedup、quality、sampling、scene、pipeline）
- 数据资产层：`raven_trainer/data_asset/`（annotation / changelog / feedback / job / lineage / metadata / quality / recipe / registry / split / statistics / tag / version / workspace）
- 训练与导出：`raven_trainer/adapters/`（backend adapter 接口）、`trainer` 相关 + `exporters/`（ONNX 导出、`onnx_verify.py`）、`scripts/export_obb_onnx.py`
- API：`raven_trainer/api/`（FastAPI `app`，前缀 `/api/v1`，routers 按领域拆分：datasets / training / models / job / ingestion / reports / …）
- 其他：`cli/`（`raven-train` 命令）、`augment/`、`callbacks/`、`metrics/`、`plugins/`、`configs/`（YAML 模板）、`templates/`、`integration/`（raven_go.py）
- 业务项目：`projects/`（仅配置，无业务代码）；数据存储统一放 `data/`；`ui_src/` React 前端
- **不要修改**：`data/`、`.venv/`、`__pycache__/`、`*.egg-info/`、`dist/`、`build/`，以及被 gitignore 的模型权重（`*.pt/*.pth/*.onnx/*.engine/*.trt`）

## Code Conventions
- 每个 Python 文件 ≤ 500 行、每个类 ≤ 300 行、每个方法 ≤ 80 行；超限拆分为子模块，禁止超大 `dataset.py`
- 导入顺序：`from __future__ import annotations` → 标准库 → 第三方 → 本地模块，分组空行分隔
- 命名：包 lowercase 无下划线、模块 snake_case、类 PascalCase、函数/变量 snake_case、常量 UPPER_CASE、配置 key snake_case
- 所有公共类 / 方法必须有 docstring（Google 风格）与完整类型注解
- 配置用 Pydantic 模型 + YAML（`configs/` / `templates/`）；日志用 loguru，不用裸 print 刷日志
- 插件开发：每个插件放 `plugins/<name>/`，提供 `setup()`，可注册 adapter / augmentation / exporter，**不得修改核心平台代码**
- 训练适配器作为仓库内 dapters/ 下的独立可分发包（raven-ultralytics 等），核心包不直接内嵌模型框架
- **Dataset Recipe 是数据集的“源代码”**：维护 recipe 而非数据本身，保证同一 recipe 产出同一数据集
- OBB 模型导出一律用 `scripts/export_obb_onnx.py` 包装脚本（统一 family imgsz / 内嵌 NMS / metadata），与 raven-go、raven-onnxruntime 的 OBB 约定对齐

## Testing
- 测试框架 pytest（配置在 `pyproject.toml`，`testpaths = ["tests"]`）；新增功能 / Bug 修复必须包含对应测试
- 纯逻辑（管线、转换器、统计、recipe/workspace 逻辑、导出元数据）优先写不依赖 GPU / ONNX 的单元测试
- 依赖真实模型 / 训练 / 数据集的测试按集成测试处理：用 marker / 环境变量 gating，缺少数据或 GPU 时跳过而非无条件失败
- 外部服务（raven-go、训练后端、网络下载）必须 mock，禁止测试里的真实请求
- 提交前保证 `uv run pytest` 通过，并确保 `uv run pylint` + `uv run mypy` 通过

## Boundaries（禁止事项）
- 禁止自动执行 `git push`、删除文件、修改生产配置
- 禁止提交数据：`data/` 全部数据、`projects/`、`datasets/**/data/`、`raw_data/`、模型权重（`*.pt/*.pth/*.onnx/*.engine/*.trt`）、`.rar` 打包文件、`.venv/`、`__pycache__/`、`*.egg-info/`
- 禁止未经确认安装 / 升级依赖（Python 依赖用 uv，`ui_src/` 用 npm，勿交叉引入 lockfile）
- 禁止大规模重构或「顺手优化」未要求的代码；不随意改动 `docs/*_SPEC.md`（它们是领域契约 / 规范）
- 规范文档多版本并存（如 `data-storage-spec-v2*.md`）：改动时先确认要跟进哪份，别想当然覆盖
- OBB 导出不要绕过 `scripts/export_obb_onnx.py` 手写导出逻辑
- 需求不清晰时先提问，不要假设

## PR & Commit
- Commit 信息使用 Conventional Commits：`feat:`, `fix:`, `chore:`, `test:`, `docs:` 等
- PR 标题简洁描述变更，描述中说明测试情况
- 改动前先阅读相关现有代码、测试与对应 `docs/*_SPEC.md` 规范；提交前自检 `uv run pytest` + `uv run pylint` + `uv run mypy`

## Security
- 所有输入校验（CLI 参数、YAML 配置、API 请求体/查询、文件与路径、数据字段）
- API 敏感操作需权限 / 鉴权检查；上传 / 导入数据需校验来源与类型
- 永不在日志中输出密钥、token、数据集原始内容或用户隐私（日志走 loguru）
- 模型 / 数据来自可信来源，导出元数据校验 `sha256` 与实际文件一致
