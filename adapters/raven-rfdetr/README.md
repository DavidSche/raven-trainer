# raven-rfdetr

Raven Trainer 的 RF-DETR（detection / segmentation）后端适配器。

通过 entry_points 组 `raven_trainer.adapters` 注册，backend name = `rfdetr`。
不改动 `rf-detr` / `rf-detr-onnx` 上游源码（rf-detr-onnx 仅作只读导出参考）。

## 安装

```bash
# 1. 锁定版本基线（P0-1）：本地源码 1.10.0.dev
pip install -e E:/study-place/Davidche/2026/github/rf-detr
# 2. 训练依赖（PTL 排除 2.6.2/2.6.3）
pip install "rfdetr[train,loggers]"
# 3. 本包（含 raven-trainer 平台层）
pip install -e E:/study-place/Davidche/2026/raven/raven-rfdetr
```

建议在独立 venv 中安装，避免与 raven-trainer / raven-ultralytics 的
PyTorch 版本互相干扰。

## 使用

切换 backend 只需修改 `project.yaml`：

```yaml
backend:
  name: rfdetr            # 原 ultralytics
  model: rf-detr-medium   # 原 yolo26s
```

训练 / 评估 / 导出命令不变：

```bash
raven-train train   projects/<name>/project.yaml
raven-train eval    projects/<name>/project.yaml
raven-train export  projects/<name>/project.yaml
```

## 支持模型

| Raven model | rf-detr 类 | 默认 resolution |
|-------------|-----------|----------------|
| `rf-detr-nano` | `RFDETRNano` | 384 |
| `rf-detr-small` | `RFDETRSmall` | 512 |
| `rf-detr-base`（deprecated） | `RFDETRBase` | 560 |
| `rf-detr-medium` | `RFDETRMedium` | 576 |
| `rf-detr-large` | `RFDETRLarge` | 704 |
| `rf-detr-seg-nano/small/medium/large` | `RFDETRSeg*` | 312/384/432/504 |

## 关键行为

- `train()`：`progress_bar="tqdm"` + 后台线程 tail `metrics.csv` 上报
  `metric_func`；返回 `{model_path, run_dir, metrics}`（上游返回 None）。
- `evaluate()`：指标键前缀剥离 + 显式映射（`val/mAP_50_95` → `mAP50-95`，
  `mAR` 无后缀），空指标会令 CLI 判 FAILED，故必须正确归一化。
- `export()`：vendor 自 rf-detr-onnx 的导出流程（局部 antialias patch，
  try/finally 恢复），自研 metadata（model_name/version/labels/sha256/
  resolution/归一化常量）与 pt-vs-onnx verify。
- `data.yaml`：每次训练/评估前从 Raven `dataset.yaml` 幂等重生成，
  剔除 `test` 字段、`path` 用绝对 posix 路径（Windows 安全）。

## 测试

```bash
pip install -e ".[dev]"
pytest
```
