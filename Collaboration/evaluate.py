"""
协同推理评估脚本 - 对比 VLM / YOLO / 协同推理 的判断精度
二分类：unsafe（不安全）/ safe（安全）

用法：
    python evaluate.py --images ./images --labels ./labels
    python evaluate.py --images ./images --labels ./labels --output ./results.csv
    python evaluate.py --images ./images --labels ./labels --no-grid-check
"""
import argparse
import csv
import os
import re
import time
import torch
import numpy as np
from PIL import Image

# ======================== 配置 ========================

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_ADAPTER_PATH = "./model/qwen2.5-vl-fire"
YOLO_MODEL_PATH = "./model/yolo11-fire/best.onnx"

YOLO_CLASS_NAMES = {0: "fire", 1: "smoke"}
YOLO_CONF_HIGH = 0.5
YOLO_CONF_LOW = 0.2

GRID_BOOST_PER_MATCH = 0.1
GRID_BOOST_MAX = 0.9

VLM_PROMPT = "请先判断是否为火灾图像。如果是，请说明火焰占据图片九宫格的哪几个版块；若不是，则直接回答不是火灾图像即可，无需分析版块。"

GRID_NAMES = {
    "左上": (0, 0), "中上": (0, 1), "右上": (0, 2),
    "左中": (1, 0), "中":   (1, 1), "右中": (1, 2),
    "左下": (2, 0), "中下": (2, 1), "右下": (2, 2),
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ======================== 模型加载 ========================

def load_models():
    """加载 VLM 和 YOLO 模型"""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from ultralytics import YOLO

    print("加载 Qwen2.5-VL 基座模型...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    print("加载 LoRA adapter...")
    qwen_model = PeftModel.from_pretrained(base_model, QWEN_ADAPTER_PATH)
    qwen_model.eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME,
        min_pixels=224 * 224,
        max_pixels=1280 * 1280,
    )

    print("加载 YOLO 模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("模型加载完成。\n")
    return qwen_model, processor, yolo_model


# ======================== VLM 推理 ========================

def vlm_infer(image_path, qwen_model, processor):
    """
    VLM 推理，返回 (output_text, is_fire, grid_cells)
    使用 validate_new.py 的判别方式
    """
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path,
             "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": VLM_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(qwen_model.device)

    try:
        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **inputs, max_new_tokens=128, min_new_tokens=20
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    finally:
        del inputs
        if "generated_ids" in locals():
            del generated_ids
        torch.cuda.empty_cache()

    is_fire = extract_fire_decision(output_text)
    grid_cells = (
        [name for name in GRID_NAMES if name in output_text]
        if is_fire else []
    )
    return output_text, is_fire, grid_cells


def extract_fire_decision(text):
    """
    提取火灾判断（validate_new.py 方式）
    返回 True(火灾) / False(非火灾) / None(无法判断)
    """
    text = text.strip()
    if "不" in text and "火灾" in text:
        not_pos = text.find("不")
        fire_pos = text.find("火灾")
        if not_pos < fire_pos and fire_pos - not_pos < 20:
            return False
    if "是" in text and "火灾" in text:
        is_pos = text.find("是")
        fire_pos = text.find("火灾")
        if is_pos < fire_pos and fire_pos - is_pos < 20:
            return True
    return None


# ======================== YOLO 推理 ========================

def yolo_infer(image_path, yolo_model, conf_threshold):
    """
    YOLO 检测，返回检测结果列表
    每个元素: {"bbox": [x1,y1,x2,y2], "conf": float, "class": str}
    """
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        names = (
            r.names
            if r.names and not all(str(v).isdigit() for v in r.names.values())
            else YOLO_CLASS_NAMES
        )
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "conf": float(box.conf[0]),
                "class": names.get(cls_id, str(cls_id)),
            })
    return detections


# ======================== 位置一致性检查 ========================

def bbox_to_grids(bbox, img_w, img_h):
    """将检测框映射到九宫格板块"""
    x1, y1, x2, y2 = bbox
    col_w, row_h = img_w / 3, img_h / 3
    grids = []
    for name, (row, col) in GRID_NAMES.items():
        gx1, gy1 = col * col_w, row * row_h
        gx2, gy2 = gx1 + col_w, gy1 + row_h
        if x1 < gx2 and x2 > gx1 and y1 < gy2 and y2 > gy1:
            grids.append(name)
    return grids


def grid_consistency_check(vlm_grids, detections, img_w, img_h):
    """位置一致性检查，重合板块提升置信度"""
    if not vlm_grids:
        return detections
    vlm_grid_set = set(vlm_grids)
    enhanced = []
    for det in detections:
        det_grids = bbox_to_grids(det["bbox"], img_w, img_h)
        overlap = vlm_grid_set & set(det_grids)
        boost = len(overlap) * GRID_BOOST_PER_MATCH
        new_conf = min(det["conf"] + boost, GRID_BOOST_MAX)
        enhanced.append({
            **det,
            "conf": new_conf,
            "original_conf": det["conf"],
            "grid_overlap": list(overlap),
            "position_consistent": len(overlap) > 0,
        })
    return enhanced


# ======================== 协同推理预警 ========================

def classify_alert(vlm_is_fire, yolo_has_fire_smoke):
    """四级预警"""
    if vlm_is_fire and yolo_has_fire_smoke:
        return "ALARM"
    elif not vlm_is_fire and yolo_has_fire_smoke:
        return "CAUTION"
    elif vlm_is_fire and not yolo_has_fire_smoke:
        return "ATTENTION"
    else:
        return "SAFE"


# ======================== 数据加载 ========================

def load_dataset(images_dir, labels_dir):
    """
    加载图片和标签，返回 [(image_path, label, category), ...]
    label: "unsafe" / "safe"
    category: 文件名前缀（fire / smoke / small / attention / nofire）
    """
    dataset = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            print(f"[警告] 标签文件不存在，跳过: {label_path}")
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.read().strip().lower()
        # 提取类别前缀
        category = re.match(r"[a-zA-Z]+", stem)
        category = category.group() if category else "unknown"
        dataset.append((os.path.join(images_dir, fname), label, category))
    return dataset


# ======================== 指标计算 ========================

def compute_metrics(y_true, y_pred):
    """
    计算二分类指标（正类 = unsafe）
    返回 {accuracy, precision, recall, f1}
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "unsafe" and p == "unsafe")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p == "safe")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "safe" and p == "unsafe")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "unsafe" and p == "safe")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ======================== 主评估流程 ========================

def evaluate(images_dir, labels_dir, output_csv, enable_grid_check=True):
    dataset = load_dataset(images_dir, labels_dir)
    if not dataset:
        print("未找到有效的图片-标签对。")
        return
    print(f"共加载 {len(dataset)} 张图片\n")

    # 加载模型
    qwen_model, processor, yolo_model = load_models()

    # 存储每张图的详细结果
    rows = []
    # 存储三个模型的预测
    vlm_preds, yolo_preds, collab_preds = [], [], []
    true_labels = []
    # 推理计时
    vlm_total_time = 0.0
    yolo_total_time = 0.0
    collab_total_time = 0.0
    # 协同推理细分统计
    alert_counts = {"ALARM": 0, "CAUTION": 0, "ATTENTION": 0, "SAFE": 0}
    # YOLO 检测类型分布
    yolo_det_types = {"fire_only": 0, "smoke_only": 0, "both": 0, "nothing": 0}

    for idx, (img_path, label, category) in enumerate(dataset):
        img = Image.open(img_path)
        img_w, img_h = img.size
        img.close()
        basename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(dataset)}] {basename} (真实标签: {label})")

        # ---- 纯 VLM 推理 ----
        t0 = time.time()
        vlm_text, vlm_is_fire, vlm_grids = vlm_infer(
            img_path, qwen_model, processor
        )
        vlm_time = time.time() - t0
        vlm_total_time += vlm_time

        # VLM 判断：True/None → unsafe，False → safe
        if vlm_is_fire is False:
            vlm_pred = "safe"
        else:
            vlm_pred = "unsafe"

        # ---- 纯 YOLO 推理（固定阈值 0.5）----
        t0 = time.time()
        yolo_dets_standalone = yolo_infer(img_path, yolo_model, YOLO_CONF_HIGH)
        yolo_time = time.time() - t0
        yolo_total_time += yolo_time

        yolo_has_det = len(yolo_dets_standalone) > 0
        yolo_pred = "unsafe" if yolo_has_det else "safe"

        # YOLO 检测类型分布
        yolo_classes = set(d["class"] for d in yolo_dets_standalone)
        has_fire = "fire" in yolo_classes
        has_smoke = "smoke" in yolo_classes
        if has_fire and has_smoke:
            yolo_det_types["both"] += 1
            yolo_det_type = "fire+smoke"
        elif has_fire:
            yolo_det_types["fire_only"] += 1
            yolo_det_type = "fire_only"
        elif has_smoke:
            yolo_det_types["smoke_only"] += 1
            yolo_det_type = "smoke_only"
        else:
            yolo_det_types["nothing"] += 1
            yolo_det_type = "nothing"

        # ---- 协同推理 ----
        # VLM 结果已有，只需额外做 YOLO（动态阈值）+ 位置检查 + 预警
        t0 = time.time()
        collab_conf = YOLO_CONF_LOW if vlm_is_fire else YOLO_CONF_HIGH
        collab_dets = yolo_infer(img_path, yolo_model, collab_conf)

        if enable_grid_check and vlm_grids and collab_dets:
            collab_dets = grid_consistency_check(
                vlm_grids, collab_dets, img_w, img_h
            )

        collab_yolo_has = len(collab_dets) > 0
        # 协同推理中 VLM 判断：is_fire 为 True 或 None 都视为"VLM认为火灾"
        collab_vlm_fire = vlm_is_fire is not False
        alert_level = classify_alert(collab_vlm_fire, collab_yolo_has)
        collab_time = vlm_time + (time.time() - t0)  # VLM 时间 + 后续处理时间
        collab_total_time += collab_time

        collab_pred = "safe" if alert_level == "SAFE" else "unsafe"
        alert_counts[alert_level] += 1

        # 记录
        true_labels.append(label)
        vlm_preds.append(vlm_pred)
        yolo_preds.append(yolo_pred)
        collab_preds.append(collab_pred)

        rows.append({
            "image": basename,
            "category": category,
            "true_label": label,
            "vlm_output": vlm_text,
            "vlm_is_fire": vlm_is_fire,
            "vlm_pred": vlm_pred,
            "vlm_time": f"{vlm_time:.2f}",
            "yolo_det_count": len(yolo_dets_standalone),
            "yolo_det_type": yolo_det_type,
            "yolo_pred": yolo_pred,
            "yolo_time": f"{yolo_time:.2f}",
            "collab_alert": alert_level,
            "collab_pred": collab_pred,
            "collab_time": f"{collab_time:.2f}",
        })

        print(f"  VLM: {vlm_pred}({vlm_time:.2f}s) | "
              f"YOLO: {yolo_pred}({yolo_time:.2f}s) | "
              f"协同: {collab_pred}[{alert_level}]({collab_time:.2f}s)")

    # ======================== 计算指标 ========================
    vlm_metrics = compute_metrics(true_labels, vlm_preds)
    yolo_metrics = compute_metrics(true_labels, yolo_preds)
    collab_metrics = compute_metrics(true_labels, collab_preds)

    # ======================== 输出结果 ========================
    n = len(dataset)
    print(f"\n{'='*70}")
    print(f"评估结果（共 {n} 张图片）")
    print(f"{'='*70}")

    # 主表格
    header = f"{'Model':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Avg Time':>10}"
    print(header)
    print("-" * 70)
    for name, m, total_t in [
        ("VLM", vlm_metrics, vlm_total_time),
        ("YOLO", yolo_metrics, yolo_total_time),
        ("Collaborative", collab_metrics, collab_total_time),
    ]:
        avg_t = total_t / n if n > 0 else 0
        print(f"{name:<16} {m['accuracy']:>9.1%} {m['precision']:>10.1%} "
              f"{m['recall']:>10.1%} {m['f1']:>10.3f} {avg_t:>9.2f}s")

    # 协同推理细分统计
    print(f"\n{'='*70}")
    print("协同推理预警分布")
    print(f"{'='*70}")
    for level in ["ALARM", "CAUTION", "ATTENTION", "SAFE"]:
        print(f"  {level:<12} {alert_counts[level]:>3} 张")

    # YOLO 检测类型分布
    print(f"\n{'='*70}")
    print("YOLO 检测类型分布（纯 YOLO，阈值 0.5）")
    print(f"{'='*70}")
    for dtype, count in yolo_det_types.items():
        print(f"  {dtype:<12} {count:>3} 张")

    # 推理总时间
    print(f"\n{'='*70}")
    print("推理时间统计")
    print(f"{'='*70}")
    for name, total_t in [
        ("VLM", vlm_total_time),
        ("YOLO", yolo_total_time),
        ("Collaborative", collab_total_time),
    ]:
        avg_t = total_t / n if n > 0 else 0
        print(f"  {name:<16} 总计: {total_t:.1f}s  平均: {avg_t:.2f}s/张")

    # ======================== 保存详细 CSV ========================
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n详细结果已保存: {output_csv}")


# ======================== 命令行入口 ========================

def main():
    parser = argparse.ArgumentParser(
        description="VLM / YOLO / 协同推理 判断精度对比评估"
    )
    parser.add_argument(
        "--images", type=str, default="./images", help="图片目录路径"
    )
    parser.add_argument(
        "--labels", type=str, default="./labels", help="标签目录路径"
    )
    parser.add_argument(
        "--output", type=str, default="./evaluation_results.csv",
        help="详细结果 CSV 保存路径",
    )
    parser.add_argument(
        "--no-grid-check", action="store_true", help="禁用位置一致性检查"
    )
    args = parser.parse_args()

    evaluate(
        images_dir=args.images,
        labels_dir=args.labels,
        output_csv=args.output,
        enable_grid_check=not args.no_grid_check,
    )


if __name__ == "__main__":
    main()
