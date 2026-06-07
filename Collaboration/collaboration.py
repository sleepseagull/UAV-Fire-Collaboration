"""
协同推理系统 - VLM + YOLO 火灾检测
流程：VLM判断 → 动态调整YOLO阈值 → YOLO检测 → 位置一致性检查 → 分级预警

用法：
    python collaboration.py --image path/to/image.jpg
    python collaboration.py --image path/to/image.jpg --no-grid-check
    python collaboration.py --image-dir path/to/images/
    python collaboration.py --image path/to/image.jpg --save-vis
    python collaboration.py --image-dir path/to/images/ --save-vis --output-dir ./results
"""
import argparse
import os
import re
import torch
import numpy as np
import cv2
import supervision as sv
from PIL import Image, ImageFont, ImageDraw

# ======================== 配置 ========================

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_ADAPTER_PATH = "./model/qwen2.5-vl-fire"
YOLO_MODEL_PATH = "./model/yolo11-fire/best.onnx"

# YOLO 类名映射（ONNX 模型可能不包含类名，需要手动指定）
YOLO_CLASS_NAMES = {0: "fire", 1: "smoke"}

# YOLO 阈值
YOLO_CONF_HIGH = 0.5   # VLM 未检测到火灾时的阈值
YOLO_CONF_LOW = 0.2    # VLM 检测到火灾时的阈值

# 位置一致性增强
GRID_BOOST_PER_MATCH = 0.1  # 每重合一个板块提升的置信度
GRID_BOOST_MAX = 0.9        # 置信度上限

# VLM 提示词
VLM_PROMPT = "请先判断是否为火灾图像。如果是，请说明火焰占据图片九宫格的哪几个版块；若不是，则直接回答不是火灾图像即可，无需分析版块。"

# 九宫格板块名称映射
GRID_NAMES = {
    "左上": (0, 0), "中上": (0, 1), "右上": (0, 2),
    "左中": (1, 0), "中": (1, 1), "右中": (1, 2),
    "左下": (2, 0), "中下": (2, 1), "右下": (2, 2),
}

# 可视化配置
ALERT_COLORS = {
    "ALARM":     (0, 0, 255),     # 红色 (BGR)
    "CAUTION":   (0, 165, 255),   # 橙色
    "ATTENTION": (0, 255, 255),   # 黄色
    "SAFE":      (0, 200, 0),     # 绿色
}
VLM_GRID_COLOR = (0, 0, 255)     # 红色半透明填充 (BGR)
VLM_GRID_ALPHA = 0.25
GRID_LINE_COLOR = (200, 200, 200) # 浅灰网格线

# ======================== 模型加载 ========================

def load_models():
    """加载 VLM 和 YOLO 模型，返回 (qwen_model, processor, yolo_model)"""
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

def vlm_analyze(image_path, qwen_model, processor):
    """
    用 VLM 分析图片，返回 (is_fire: bool, grid_cells: list[str])
    grid_cells 示例: ["左上", "中上"]
    """
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": VLM_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(qwen_model.device)

    try:
        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=128, min_new_tokens=20)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    finally:
        del inputs
        if 'generated_ids' in locals():
            del generated_ids
        torch.cuda.empty_cache()

    print(f"[VLM 输出] {output_text}")

    # 解析结果
    is_fire = not bool(re.search(r"不是.{0,5}火灾", output_text))
    grid_cells = [name for name in GRID_NAMES if name in output_text] if is_fire else []

    return is_fire, grid_cells, output_text


# ======================== YOLO 推理 ========================

def yolo_detect(image_path, yolo_model, conf_threshold):
    """
    用 YOLO 检测图片，返回检测结果列表
    每个元素: {"bbox": [x1,y1,x2,y2], "conf": float, "class": str}
    """
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)
    detections = []
    for r in results:
        names = r.names if r.names and not all(str(v).isdigit() for v in r.names.values()) else YOLO_CLASS_NAMES
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                "conf": float(box.conf[0]),
                "class": names.get(cls_id, str(cls_id)),
            })
    return detections


# ======================== 位置一致性检查 ========================

def bbox_to_grids(bbox, img_w, img_h):
    """将 YOLO 检测框映射到九宫格板块，返回板块名称列表"""
    x1, y1, x2, y2 = bbox
    col_w, row_h = img_w / 3, img_h / 3
    grids = []
    for name, (row, col) in GRID_NAMES.items():
        gx1, gy1 = col * col_w, row * row_h
        gx2, gy2 = gx1 + col_w, gy1 + row_h
        # 检测框与网格有交集
        if x1 < gx2 and x2 > gx1 and y1 < gy2 and y2 > gy1:
            grids.append(name)
    return grids


def grid_consistency_check(vlm_grids, detections, img_w, img_h):
    """
    位置一致性检查：对比 VLM 板块和 YOLO 检测框板块
    如果有重合，提升该检测的置信度（每重合一个板块 +0.1，上限 0.9）
    返回增强后的 detections
    """
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


# ======================== 双重验证与预警 ========================

def classify_alert(vlm_is_fire, yolo_has_fire_smoke):
    """
    四级预警：
    1. VLM=火灾 + YOLO=有火/烟 → 警报
    2. VLM=非火灾 + YOLO=有火/烟 → 注意防火
    3. VLM=火灾 + YOLO=无火/烟 → 需要人为关注
    4. VLM=非火灾 + YOLO=无火/烟 → 安全
    """
    if vlm_is_fire and yolo_has_fire_smoke:
        return "ALARM", "触发警报：VLM 和 YOLO 均检测到火灾/烟雾"
    elif not vlm_is_fire and yolo_has_fire_smoke:
        return "CAUTION", "注意防火：VLM 未检测到火灾，但 YOLO 检测到火/烟"
    elif vlm_is_fire and not yolo_has_fire_smoke:
        return "ATTENTION", "需要人为关注：VLM 判断为火灾，但 YOLO 未检测到火/烟"
    else:
        return "SAFE", "安全：VLM 和 YOLO 均未检测到火灾"


# ======================== 可视化 ========================


def _find_chinese_font(size=16):
    """查找系统中可用的中文字体"""
    font_paths = [
        # Windows
        "msyh.ttc", "simhei.ttf", "simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        # Linux (AutoDL / Ubuntu)
        "/usr/share/fonts/chinese/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size)
            print(f"[Font] 使用字体: {path}")
            return font
        except (OSError, IOError):
            continue
    print("[Font] 警告: 未找到中文字体，使用默认字体")
    return ImageFont.load_default()


def visualize_result(result, output_path=None):
    """
    可视化协同推理结果：左侧图片(YOLO检测框+九宫格线) + 右侧命令行风格文字面板
    返回: annotated BGR numpy array
    """
    img = cv2.imread(result["image"])
    h, w = img.shape[:2]
    annotated = img.copy()

    # ---- A. VLM 九宫格网格线（仅画线，不高亮填充） ----
    cell_w, cell_h = w / 3, h / 3
    for i in range(1, 3):
        cv2.line(annotated, (int(i * cell_w), 0), (int(i * cell_w), h), GRID_LINE_COLOR, 1)
        cv2.line(annotated, (0, int(i * cell_h)), (w, int(i * cell_h)), GRID_LINE_COLOR, 1)

    # ---- B. YOLO 检测框 (supervision) ----
    detections = result["detections"]
    if detections:
        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["conf"] for d in detections], dtype=np.float32)
        class_names = [d["class"] for d in detections]
        name_to_id = {"fire": 0, "smoke": 1}
        class_id = np.array([name_to_id.get(n, 0) for n in class_names])

        sv_detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id,
        )
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(class_names, confidence)
        ]

        box_cls = getattr(sv, "BoxAnnotator", None) or sv.BoundingBoxAnnotator
        box_annotator = box_cls(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        annotated = box_annotator.annotate(scene=annotated, detections=sv_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=sv_detections, labels=labels)

    # ---- C. 右侧文字面板（复现命令行输出） ----
    lines = []
    lines.append("=" * 50)
    lines.append(f"图片: {os.path.basename(result['image'])}")
    lines.append("=" * 50)
    lines.append(f"[步骤1] VLM 分析")
    vlm_text = result['vlm_output'].replace("\\n", "\n")
    vlm_lines = vlm_text.split("\n")
    lines.append(f"[VLM 输出] {vlm_lines[0]}")
    for vl in vlm_lines[1:]:
        lines.append(f"  {vl}")
    vlm_label = "火灾" if result["vlm_is_fire"] else "非火灾"
    lines.append(f"  VLM 判断: {vlm_label}")
    if result["vlm_grids"]:
        lines.append(f"  火焰板块: {result['vlm_grids']}")
    lines.append("")
    lines.append(f"[步骤2] YOLO 置信度阈值: {result['yolo_conf_threshold']}")
    lines.append(f"[步骤3] YOLO 检测")
    lines.append(f"  检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        bbox_str = [round(v, 1) for v in det["bbox"]]
        line = f"    [{i+1}] {det['class']} 置信度={det['conf']:.3f} 框={bbox_str}"
        lines.append(line)
    # 位置一致性
    consistent_dets = [d for d in detections if d.get("position_consistent")]
    if consistent_dets:
        lines.append("")
        lines.append(f"[步骤4] 位置一致性检查")
        for i, det in enumerate(consistent_dets):
            lines.append(f"    {det['class']} 置信度 {det['original_conf']:.3f} → {det['conf']:.3f} (重合板块: {det['grid_overlap']})")
    lines.append("")
    lines.append(f"[结果] {result['alert_level']}: {result['alert_message']}")

    # 计算面板尺寸
    font_size = max(12, min(18, h // 35))
    font = _find_chinese_font(font_size)
    line_height = font_size + 10

    # 自动换行：中文字符按2倍宽度计算
    panel_w = max(w, 500)  # 面板至少 500px 宽
    char_width = font_size // 2 + 1
    max_width = panel_w - 20  # 左右各留10px边距
    wrapped_lines = []
    for line in lines:
        cur_w = 0
        cur_line = ""
        for ch in line:
            cw = char_width * 2 if ord(ch) > 127 else char_width
            if cur_w + cw > max_width and cur_line:
                wrapped_lines.append(cur_line)
                cur_line = "  " + ch
                cur_w = char_width * 2 + cw
            else:
                cur_line += ch
                cur_w += cw
        if cur_line:
            wrapped_lines.append(cur_line)

    panel_h = max(h, line_height * len(wrapped_lines) + 20)

    # 创建面板并用 PIL 绘制
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 30  # 深灰背景
    pil_panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_panel)

    alert_color_bgr = ALERT_COLORS.get(result["alert_level"], (255, 255, 255))
    alert_color_rgb = (alert_color_bgr[2], alert_color_bgr[1], alert_color_bgr[0])

    y = 10
    for line in wrapped_lines:
        # 结果行用预警颜色
        if line.startswith("[结果]"):
            text_color = alert_color_rgb
        elif line.startswith("="):
            text_color = (100, 100, 100)
        else:
            text_color = (220, 220, 220)
        draw.text((10, y), line, font=font, fill=text_color)
        y += line_height

    panel = cv2.cvtColor(np.array(pil_panel), cv2.COLOR_RGB2BGR)

    # 高度对齐：让图片和面板等高
    if panel_h > h:
        pad = np.zeros((panel_h - h, w, 3), dtype=np.uint8)
        annotated = np.vstack([annotated, pad])
    elif h > panel_h:
        pad = np.zeros((h - panel_h, panel_w, 3), dtype=np.uint8) + 30
        panel = np.vstack([panel, pad])

    # 拼接：图片在左，面板在右
    annotated = np.hstack([annotated, panel])

    # ---- D. 保存 ----
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, annotated)
        print(f"[可视化] 已保存: {output_path}")

    return annotated


# ======================== 协同推理主流程 ========================

def collaborative_infer(image_path, qwen_model, processor, yolo_model, enable_grid_check=True):
    """
    协同推理主函数
    返回: {
        "image": 图片路径,
        "vlm_is_fire": bool,
        "vlm_grids": list,
        "vlm_output": str,
        "yolo_conf_threshold": float,
        "detections": list,
        "alert_level": str,
        "alert_message": str,
    }
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    # 第1步：VLM 判断
    print(f"\n{'='*50}")
    print(f"图片: {image_path}")
    print(f"{'='*50}")
    print("[步骤1] VLM 分析中...")
    vlm_is_fire, vlm_grids, vlm_output = vlm_analyze(image_path, qwen_model, processor)
    print(f"  VLM 判断: {'火灾' if vlm_is_fire else '非火灾'}")
    if vlm_grids:
        print(f"  火焰板块: {vlm_grids}")

    # 第2步：动态调整 YOLO 阈值
    conf_threshold = YOLO_CONF_LOW if vlm_is_fire else YOLO_CONF_HIGH
    print(f"\n[步骤2] YOLO 置信度阈值: {conf_threshold}")

    # 第3步：YOLO 检测
    print("[步骤3] YOLO 检测中...")
    detections = yolo_detect(image_path, yolo_model, conf_threshold)
    print(f"  检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        print(f"    [{i+1}] {det['class']} 置信度={det['conf']:.3f} 框={[round(v,1) for v in det['bbox']]}")

    # 第4步：位置一致性检查（可选）
    if enable_grid_check and vlm_grids and detections:
        print(f"\n[步骤4] 位置一致性检查...")
        detections = grid_consistency_check(vlm_grids, detections, img_w, img_h)
        for i, det in enumerate(detections):
            if det.get("position_consistent"):
                print(f"    [{i+1}] {det['class']} 置信度 {det['original_conf']:.3f} → {det['conf']:.3f} (重合板块: {det['grid_overlap']})")

    # 第5步：双重验证
    yolo_has_fire_smoke = len(detections) > 0
    alert_level, alert_message = classify_alert(vlm_is_fire, yolo_has_fire_smoke)
    print(f"\n[结果] {alert_level}: {alert_message}")

    return {
        "image": image_path,
        "vlm_is_fire": vlm_is_fire,
        "vlm_grids": vlm_grids,
        "vlm_output": vlm_output,
        "yolo_conf_threshold": conf_threshold,
        "detections": detections,
        "alert_level": alert_level,
        "alert_message": alert_message,
    }


# ======================== 命令行入口 ========================

def main():
    parser = argparse.ArgumentParser(description="VLM + YOLO 协同火灾检测")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image-dir", type=str, help="图片目录路径（批量推理）")
    parser.add_argument("--no-grid-check", action="store_true", help="禁用位置一致性检查")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化结果图片")
    parser.add_argument("--output-dir", type=str, default="./output", help="可视化结果保存目录")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("请指定 --image 或 --image-dir")

    # 收集图片路径
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("未找到图片文件。")
        return

    # 加载模型（只加载一次）
    qwen_model, processor, yolo_model = load_models()

    # 逐张推理
    results = []
    for img_path in image_paths:
        result = collaborative_infer(
            img_path, qwen_model, processor, yolo_model,
            enable_grid_check=not args.no_grid_check,
        )
        results.append(result)

        if args.save_vis:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(args.output_dir, f"{basename}_vis.jpg")
            visualize_result(result, output_path=output_path)

    # 汇总
    if len(results) > 1:
        print(f"\n{'='*50}")
        print(f"批量推理汇总（共 {len(results)} 张）")
        print(f"{'='*50}")
        for r in results:
            print(f"  [{r['alert_level']}] {os.path.basename(r['image'])}")


if __name__ == "__main__":
    main()
