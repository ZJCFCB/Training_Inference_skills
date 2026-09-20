#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训推一致性 dump 语义边界精确比对脚本(架构无关)。

对两侧已采集的 dump(training/prefill vs inference/prefill)按语义执行顺序逐边界跑精确门禁:
  same_shape / same_dtype / no_nonfinite / torch.equal / max_abs_diff,
沿执行顺序定位第一个 NOT_EXACT 边界(首差异),输出机器可读 JSON + 终端摘要。

用法:
  python3 analyze_dump_boundaries.py \
      --train <train>/step0/rank0/dump_tensor_data \
      --infer <infer>/step0/rank0/dump_tensor_data \
      --mapping semantic_mapping.json \
      --output boundary_results.json

语义映射 JSON(list of BOUNDARY,格式见下文 schema)由分析者按现场 dump 的算子名/形状
自建——不绑定任何训练/推理框架或模型;两侧算子命名差异极大,必须按数学语义对齐。

只读:dump 数据只读,不修改任何 dump 文件。
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import sys
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# 语义映射 schema
# ---------------------------------------------------------------------------
# BOUNDARY = {
#   "name": "唯一名(结果 key)",
#   "semantic_boundary": "语义边界描述,如 decoder.layer0.attention.q_down",
#   "layer": 0,                       # 可选,层号
#   "execution_order": 10,            # 可选,语义执行顺序(用于排序)
#   "train_glob": "glob 匹配训练侧 dump_tensor_data 内文件",
#   "infer_glob": "glob 匹配推理侧 dump_tensor_data 内文件",
#   "infer_globs": [...],             # 可选,推理侧多个 glob(逐 position 输出拼接时用,配合 infer_concat)
#   "transform": {                     # 两侧文件载入后的归一化/切片(可选)
#     "squeeze": [0,1],                # 逐个 squeeze 的 dim(允许 None=省略)
#     "reshape": [seq, hidden],        # reshape 到该 shape
#     "train_slice": "[:, :h]",        # 训练侧张量切片(exec)
#     "infer_slice": "[:, :h]",        # 推理侧张量切片(exec)
#     "infer_concat": true,            # 推理侧多 glob 的输出按 dim0 拼接
#   },
#   "note": "映射依据/备注",
# }


def load_pt(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def find_one(directory: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    return hits[0] if hits else None


def apply_transform(t: torch.Tensor, transform: dict[str, Any] | None) -> torch.Tensor:
    if not transform:
        return t
    for dim in sorted(transform.get("squeeze", []), reverse=True):
        if dim is not None and dim < t.dim() and t.shape[dim] == 1:
            t = t.squeeze(dim)
    shape = transform.get("reshape")
    if shape:
        t = t.reshape(shape)
    return t


def apply_slice(t: torch.Tensor, spec: Any) -> torch.Tensor:
    if not spec:
        return t
    return eval("t" + spec, {"t": t})  # 仅接受白名单切片表达式


def compare_boundary(
    train_dir: str, infer_dir: str, boundary: dict[str, Any]
) -> dict[str, Any]:
    name = boundary["name"]
    result: dict[str, Any] = {
        "name": name,
        "semantic_boundary": boundary.get("semantic_boundary", name),
        "classification": "MISSING",
        "evidence": [],
    }
    if "execution_order" in boundary:
        result["execution_order"] = boundary["execution_order"]
    tfile = find_one(train_dir, boundary["train_glob"])
    ifiles = [find_one(infer_dir, g) for g in boundary.get("infer_globs", [boundary["infer_glob"]])]
    ifile = ifiles[0] if ifiles else None
    if tfile is None or any(f is None for f in ifiles):
        result["classification"] = "MISSING"
        result["evidence"] = {
            "train_found": tfile is not None,
            "infer_found": any(f is not None for f in ifiles),
            "train_glob": boundary["train_glob"],
            "infer_glob": boundary.get("infer_glob", boundary.get("infer_globs")),
        }
        return result

    transform = boundary.get("transform") or {}
    t = load_pt(tfile)
    t = apply_transform(t, transform)
    t = apply_slice(t, transform.get("train_slice"))
    if len(ifiles) > 1 and transform.get("infer_concat"):
        parts = []
        for f in ifiles:
            p = apply_transform(load_pt(f), transform)
            p = apply_slice(p, transform.get("infer_slice"))
            parts.append(p)
        i = torch.cat(parts, dim=0)
    else:
        i = load_pt(ifiles[0])
        i = apply_transform(i, transform)
        i = apply_slice(i, transform.get("infer_slice"))

    result["train_file"] = os.path.relpath(tfile, train_dir)
    result["infer_file"] = os.path.relpath(ifile, infer_dir)
    result["train_shape"] = list(t.shape)
    result["infer_shape"] = list(i.shape)
    result["train_dtype"] = str(t.dtype)
    result["infer_dtype"] = str(i.dtype)

    same_shape = t.shape == i.shape
    same_dtype = t.dtype == i.dtype
    result["same_shape"] = same_shape
    result["same_dtype"] = same_dtype

    if not same_shape or not same_dtype:
        result["classification"] = "SHAPE_DIFF" if not same_shape else "DTYPE_DIFF"
        result["max_abs_diff"] = None
        result["torch_equal"] = False
        return result

    finite_t = torch.isfinite(t).all().item() if t.is_floating_point() else True
    finite_i = torch.isfinite(i).all().item() if i.is_floating_point() else True
    result["no_nonfinite"] = bool(finite_t and finite_i)

    result["torch_equal"] = bool(torch.equal(t, i))
    d = (t.float() - i.float()).abs()
    result["max_abs_diff"] = float(d.max().item())
    nz = int((d > 0).sum().item()) if same_shape else 0
    result["nz_count"] = nz
    result["nz_fraction"] = round(nz / t.numel(), 6) if t.numel() else 0.0
    # max_rel 相对差(按 train 值量级)
    denom = t.float().abs() + 1e-12
    result["max_rel_diff"] = float((d / denom).max().item())

    if not result["no_nonfinite"]:
        result["classification"] = "NONFINITE"
    elif result["torch_equal"]:
        result["classification"] = "EXACT"
    else:
        result["classification"] = "NOT_EXACT"
    return result


def run(train_dir, infer_dir, mapping, output):
    results = []
    for boundary in mapping:
        r = compare_boundary(train_dir, infer_dir, boundary)
        results.append(r)
        tag = r["classification"]
        if tag == "EXACT":
            line = f"  [EXACT] {r['name']}: {tuple(r['train_shape'])}"
        elif tag == "NOT_EXACT":
            line = (f"  [NEQ  ] {r['name']}: max_abs={r['max_abs_diff']:.3e} "
                    f"nz={r['nz_count']}/{r['nz_fraction']}")
        elif tag in ("SHAPE_DIFF", "DTYPE_DIFF"):
            line = (f"  [{tag}] {r['name']}: {r['train_shape']} vs {r['infer_shape']}")
        elif tag == "MISSING":
            line = f"  [MISS ] {r['name']}: 缺 train/infer 文件"
        else:
            line = f"  [{tag}] {r['name']}"
        print(line)

    # 汇总:沿执行顺序找首差异
    ordered = sorted(results, key=lambda r: r.get("execution_order", 0))
    summary = {
        "total": len(results),
        "exact": sum(1 for r in results if r["classification"] == "EXACT"),
        "not_exact": sum(1 for r in results if r["classification"] == "NOT_EXACT"),
        "shape_diff": sum(1 for r in results if r["classification"] == "SHAPE_DIFF"),
        "missing": sum(1 for r in results if r["classification"] == "MISSING"),
        "first_divergence": None,
        "boundaries": results,
    }
    for r in ordered:
        if r["classification"] in ("NOT_EXACT", "NONFINITE"):
            summary["first_divergence"] = {
                "name": r["name"],
                "semantic_boundary": r["semantic_boundary"],
                "max_abs_diff": r.get("max_abs_diff"),
                "nz_count": r.get("nz_count"),
            }
            break
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n结果写至 {output}")
    print("\n=== 汇总 ===")
    print("  total=%d exact=%d not_exact=%d shape_diff=%d missing=%d"
          % (summary["total"], summary["exact"], summary["not_exact"],
             summary["shape_diff"], summary["missing"]))
    fd = summary["first_divergence"]
    if fd:
        print("  首差异: %s (%s) max_abs=%.3e nz=%d"
              % (fd["name"], fd["semantic_boundary"], fd["max_abs_diff"] or 0, fd["nz_count"] or 0))
    else:
        print("  未发现首差异(覆盖范围内全部精确或缺失)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="训练侧 dump_tensor_data 目录")
    ap.add_argument("--infer", required=True, help="推理侧 dump_tensor_data 目录")
    ap.add_argument("--mapping", required=True, help="语义映射 JSON(list of BOUNDARY)")
    ap.add_argument("--output", help="结果 JSON 输出路径")
    args = ap.parse_args()
    with open(args.mapping, encoding="utf-8") as f:
        mapping = json.load(f)
    run(args.train, args.infer, mapping, args.output)


if __name__ == "__main__":
    main()
