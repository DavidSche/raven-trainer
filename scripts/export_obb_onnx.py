"""通用 OBB（旋转框）模型 ONNX 导出脚本封装。

将任意 Ultralytics OBB 权重导出为与 raven-onnxruntime 引擎兼容的 ONNX，
并自动按模型家族选择导出参数：

| 模型家族       | 引擎消费的输出                        | 导出参数              |
|----------------|---------------------------------------|-----------------------|
| yolo26*-obb    | [batch, 300, 7]（内嵌 NMS 后处理输出）| nms=True, max_det=300 |
| yolo11*-obb    | [batch, 4+nc+1, N]（原始输出）        | nms=False             |

约定（与 docs/EXPORT_SPEC.md 及 raven-onnxruntime 引擎/scanner 一致）：
- imgsz 按家族默认：yolo26=1280 / yolo11=1024（对应 Go 侧 DefaultOBBConfig 与 scanner 推断；
  静态输入模型由引擎按真实输入形状自动适配）
- opset >= 17、dynamic batch、simplify（失败自动回退为不简化）
- 元数据内嵌 model_name / model_version / labels / sha256
- 输出目录默认仓库根 models/yolo26|yolo11/（与 modelpath 常量布局一致）

用法示例：
    # 官方预训练权重（输出到仓库 models/yolo11/）
    python scripts/export_obb_onnx.py --weights yolo11m-obb.pt

    # 自定义训练的 OBB 模型
    python scripts/export_obb_onnx.py --weights runs/obb/weights/best.pt \\
        --model-name ship_obb --model-version 1.0 --labels ship,airplane

    # 仅回归校验已导出的 ONNX 是否与引擎兼容
    python scripts/export_obb_onnx.py --verify-only ../models/yolo26/yolo26m-obb.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 模型家族 → (是否内嵌 NMS, raven-onnxruntime 模型目录)
# 依据 raven-go/internal/inference/scanner.go 的家族识别与 engine_obb.go 的输出期待。
FAMILY_NMS = {
    "yolo26": True,   # yolo26 引擎直接消费 [batch, 300, 7]
    "yolo11": False,  # yolov11 引擎内置 NMS，消费 [batch, 4+nc+1, N] 原始输出
}
FAMILY_DIR = {
    "yolo26": "yolo26",
    "yolo11": "yolo11",
}
NMS_OUT_DIMS = (300, 7)  # nms 导出输出 [batch, 300, 7]
# 家族 → 默认导出 imgsz（与 Go 侧 DefaultOBBConfig.InputSize / scanner 推断一致）：
# yolo26 官方导出 imgsz=1280（动态输入）；yolo11 官方导出 imgsz=1024（静态输入）。
FAMILY_IMSZ = {
    "yolo26": 1280,
    "yolo11": 1024,
}


def infer_family(name: str) -> str:
    """从权重文件名推断 OBB 模型家族（与 scanner.go 的 containsAny 规则对齐）。"""
    base = Path(name).stem.lower()
    if "yolo26" in base or "yolov26" in base:
        return "yolo26"
    if "yolo11" in base or "yolov11" in base:
        return "yolo11"
    raise ValueError(
        f"无法从文件名推断 OBB 模型家族: {name!r}（支持 yolo26*-obb / yolo11*-obb）"
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_onnx(path: Path, family: str, num_classes: int | None = None) -> bool:
    """校验 ONNX 与对应 Go 引擎的输出期待是否匹配，并打印输入/输出/元数据。"""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def dim_str(dims: list) -> str:
        return "[" + ", ".join(str(d) for d in dims) + "]"

    inputs = [(i.name, [d.dim_value if d.dim_value > 0 else d.dim_param
                        for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
    outputs = [(o.name, [d.dim_value if d.dim_value > 0 else d.dim_param
                         for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]
    meta = {p.key: p.value for p in model.metadata_props}

    print(f"  Inputs : {[(n, dim_str(s)) for n, s in inputs]}")
    print(f"  Outputs: {[(n, dim_str(s)) for n, s in outputs]}")

    ok = True
    task = meta.get("task", "")
    if task and task != "obb":
        print(f"  [ERROR] 元数据 task={task!r}，期望 'obb'")
        ok = False

    if len(outputs) != 1:
        print(f"  [ERROR] 期望 1 个输出，实际 {len(outputs)} 个")
        return False
    shape = outputs[0][1]
    # 动态导出时部分维度为符号名（'batch'/'height'/'width'/'anchors'），仅对具体数值断言
    dim1 = shape[1] if isinstance(shape[1], int) else None
    dim2 = shape[2] if isinstance(shape[2], int) else None

    if family == "yolo26":
        # nms 输出 [batch, 300, 7] 必须为具体维度
        if len(shape) != 3 or dim1 != NMS_OUT_DIMS[0] or dim2 != NMS_OUT_DIMS[1]:
            print(f"  [ERROR] yolo26 引擎期望输出 [batch, {NMS_OUT_DIMS[0]}, {NMS_OUT_DIMS[1]}]，"
                  f"实际 {dim_str(shape)}")
            ok = False
    else:
        # 原始输出 [batch, 4+nc+1, N]（N 可为符号维度 anchors）
        if len(shape) != 3 or (dim1 is not None and dim1 < 6) or (dim2 is not None and dim2 <= 0):
            print(f"  [ERROR] yolov11 引擎期望 [batch, 4+nc+1, N] 原始输出，实际 {dim_str(shape)}")
            ok = False
        elif num_classes is not None and dim1 is not None and dim1 != 4 + num_classes + 1:
            print(f"  [WARN] 通道数 {dim1} 与 num_classes={num_classes} 不匹配"
                  f"（期望 {4 + num_classes + 1}）；自定义类别数请用 --num-classes 校正")

    family_imsz = FAMILY_IMSZ.get(family, 1024)
    for name, s in inputs:
        if len(s) == 4 and isinstance(s[2], int) and s[2] > 0 and s[2] != family_imsz:
            print(f"  [WARN] 输入 {name} 空间尺寸 {s[2]}x{s[3]}，"
                  f"Go 侧 {family} OBB 引擎/scanner 默认 {family_imsz}，请与配置核对")

    print(f"  Metadata: model_name={meta.get('model_name')!r} "
          f"model_version={meta.get('model_version')!r} "
          f"labels={meta.get('labels')!r}")
    print(f"  SHA256  : {sha256_of(path)}")
    print(f"  => {'通过' if ok else '不兼容'}（family={family}）")
    return ok


def embed_metadata(onnx_path: Path, model_name: str, model_version: str,
                   labels: list[str]) -> Path:
    """按 EXPORT_SPEC 内嵌 model_name/model_version/labels/sha256 元数据。"""
    import onnx

    model = onnx.load(str(onnx_path))
    existing = {p.key: p.value for p in model.metadata_props}
    existing.update({
        "model_name": model_name,
        "model_version": model_version,
        "labels": ",".join(labels),
    })
    del model.metadata_props[:]
    for k, v in existing.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))

    # 嵌入 sha256：先对不含该字段的成品计算校验和，再补写字段。
    # 注意：嵌入后文件二进制会变化，故命令行打印的最终 sha256 与内嵌值可能相差
    # 仅该字段本身的字节（属于预期行为）。
    stored_sha = sha256_of(onnx_path)
    model = onnx.load(str(onnx_path))
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="sha256", value=stored_sha))
    onnx.save(model, str(onnx_path))
    print(f"  sha256(embedded) = {stored_sha}")
    return onnx_path


def _resolve_family(args: argparse.Namespace, name: str) -> str | None:
    """解析家族；文件名推断失败时打印干净错误并返回 None。"""
    if args.family:
        return args.family
    try:
        return infer_family(name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return None


def run_export(args: argparse.Namespace) -> int:
    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        print(f"ERROR: 权重文件不存在: {weights}")
        return 1

    family = _resolve_family(args, weights.name)
    if family is None:
        return 1
    if family not in FAMILY_NMS:
        print(f"ERROR: 不支持的模型家族: {family}")
        return 1
    nms = FAMILY_NMS[family] if args.nms is None else args.nms
    imgsz = args.imgsz or FAMILY_IMSZ[family]
    print(f"[OBB Export] weights={weights.name} family={family} nms={nms} "
          f"imgsz={imgsz} opset={args.opset} dynamic={args.dynamic}")

    from ultralytics import YOLO  # 惰性导入：--help / --verify-only 无需 ultralytics

    model = YOLO(str(weights))
    names = getattr(model, "names", None)
    labels = args.labels or (list(names.values()) if names else [])
    if not labels:
        print("[WARN] 未获取到类别名（--labels 未指定且权重无 names），labels 元数据为空")

    model_name = args.model_name or Path(weights).stem
    out_dir = Path(args.output_dir).resolve() if args.output_dir \
        else REPO_ROOT / "models" / FAMILY_DIR[family]
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.output_name or Path(weights).stem}.onnx"

    export_kwargs = dict(
        format="onnx",
        imgsz=imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        nms=nms,
    )
    if nms:
        export_kwargs["max_det"] = 300  # 与 Go 引擎 NMS_OUT_DIMS 对齐

    try:
        exported = Path(model.export(**export_kwargs, simplify=args.simplify))
    except Exception as e:  # noqa: BLE001 - simplify 失败回退
        if not args.simplify:
            raise
        print(f"  [WARN] simplify 失败（{e}），回退为不简化重试…")
        exported = Path(model.export(**export_kwargs))

    if exported.resolve() != final_path.resolve():
        shutil.move(str(exported), str(final_path))
    print(f"  ONNX 输出: {final_path} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)")

    embed_metadata(final_path, model_name, args.model_version, labels)

    if args.copy_pt:
        shutil.copy2(weights, out_dir / weights.name)
        print(f"  PT 已复制: {out_dir / weights.name}")

    print("\n[Verify]")
    ok = verify_onnx(final_path, family, args.num_classes)
    if not ok:
        print(f"ERROR: 导出结果与 {family} 引擎不兼容，请检查参数")
        return 1
    print("\n" + "=" * 60)
    print("OBB ONNX Export Complete!")
    print("=" * 60)
    return 0


def run_verify(args: argparse.Namespace) -> int:
    path = Path(args.verify_only).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: ONNX 文件不存在: {path}")
        return 1
    family = _resolve_family(args, path.name)
    if family is None:
        return 1
    print(f"[Verify] {path} family={family}")
    ok = verify_onnx(path, family, args.num_classes)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="通用 OBB 模型 ONNX 导出封装（自动按家族匹配 raven-onnxruntime 引擎）")
    parser.add_argument("--weights", help="Ultralytics OBB 权重（*.pt），如 yolo11m-obb.pt")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入尺寸（默认按家族：yolo26=1280 / yolo11=1024，与 Go 引擎/scanner 一致）")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset（默认 17，规范要求 >=17）")
    parser.add_argument("--dynamic", action="store_true", default=True,
                        help="导出 dynamic batch（默认开启）")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="导出后 onnxslim 简化（默认开启，失败自动回退）")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--nms", dest="nms", action="store_true", default=None,
                        help="强制内嵌 NMS（默认按家族：yolo26=True / yolo11=False）")
    parser.add_argument("--no-nms", dest="nms", action="store_false")
    parser.add_argument("--family", choices=sorted(FAMILY_NMS),
                        help="覆盖家族推断（默认按文件名）")
    parser.add_argument("--output-dir", help="输出目录（默认仓库根 models/<family>/）")
    parser.add_argument("--output-name", help="输出文件名（默认与权重同名 .onnx）")
    parser.add_argument("--model-name", help="元数据 model_name（默认权重文件名）")
    parser.add_argument("--model-version", default="1.0", help="元数据 model_version（默认 1.0）")
    parser.add_argument("--labels", help="类别名列表（逗号分隔；默认取权重 names）")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="原始输出校验用的类别数（yolo11 家族，默认仅结构校验）")
    parser.add_argument("--copy-pt", action="store_true", help="同时复制 .pt 到输出目录")
    parser.add_argument("--verify-only", metavar="ONNX",
                        help="只回归校验已导出的 ONNX（不导出），家族按文件名推断")

    args = parser.parse_args()
    if args.verify_only:
        if args.weights:
            parser.error("--verify-only 与 --weights 互斥")
        return run_verify(args)
    if not args.weights:
        parser.error("导出模式需要 --weights（或用 --verify-only 校验既有 ONNX）")
    return run_export(args)


if __name__ == "__main__":
    sys.exit(main())
