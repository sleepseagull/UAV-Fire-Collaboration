"""
Qwen2.5-VL LoRA 模型验证脚本
从数据集中随机抽取样本，计算多项评估指标
"""
import json
import random
import re
import torch
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
import swanlab

# ========================= 统一配置区 =========================
# 路径配置
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"           # 基座模型路径
USE_LORA = False                                      # 是否使用 LoRA 微调模型（False 则只用基座模型）
CHECKPOINT_PATH = " "          # LoRA checkpoint 路径
ANNOTATION_FILE = "my_annotations_updated_final_prompt.json"  # 完整数据集
OUTPUT_FILE = "validation_results_lora.json" if USE_LORA else "validation_results_baseline.json"  # 验证结果保存路径

# 验证配置
NUM_SAMPLES = 100                                    # 随机抽取样本数
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 20
IMAGE_HEIGHT = 280                                   # 图片缩放高度（与训练时一致）
IMAGE_WIDTH = 280                                    # 图片缩放宽度（与训练时一致）
PROMPT = "请先判断是否为火灾图像。如果是，请说明火焰占据图片九宫格的哪几个版块；若不是，则直接回答不是火灾图像即可，无需分析版块。"

# LoRA 配置（需与训练时一致）
LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# SwanLab 配置
SWANLAB_PROJECT = "Fire-Detection-VL"
SWANLAB_EXPERIMENT = "4fire-yolo-grid-validation-lora" if USE_LORA else "4fire-yolo-grid-validation-baseline"

SEED = 42

# ========================= 九宫格位置词汇表 =========================
GRID_POSITIONS = {
    "左上", "中上", "右上",
    "左中", "中", "右中",
    "左下", "中下", "右下",
}

# ========================= 辅助函数 =========================
def extract_fire_decision(text):
    """提取火灾判断结果：True(是火灾) / False(不是火灾) / None(无法判断)"""
    text = text.strip()

    # 先检查非火灾（"不" + "火灾" 的组合）
    if "不" in text and "火灾" in text:
        # 确保"不"在"火灾"之前，避免误判如"火灾不大"
        not_pos = text.find("不")
        fire_pos = text.find("火灾")
        if not_pos < fire_pos and fire_pos - not_pos < 20:  # 距离不超过20个字符
            return False

    # 再检查是火灾（"是" + "火灾" 的组合）
    if "是" in text and "火灾" in text:
        is_pos = text.find("是")
        fire_pos = text.find("火灾")
        if is_pos < fire_pos and fire_pos - is_pos < 20:  # 距离不超过20个字符
            return True

    return None


def extract_grid_positions(text):
    """提取九宫格位置集合"""
    positions = set()
    for pos in GRID_POSITIONS:
        if pos in text:
            positions.add(pos)
    return positions


def calculate_iou(pred_set, gt_set):
    """计算集合的 IoU"""
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0
    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union


def normalize_text(text):
    """标准化文本：去除多余空格、换行，统一同义词和标点"""
    text = text.strip()
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 统一同义词
    text = text.replace("版块", "板块")
    text = text.replace("区域", "板块")
    text = text.replace("版面", "板块")
    # 统一标点符号
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("、", ",")
    # 统一"的"字
    text = text.replace("图片九宫格的", "图片九宫格")
    text = text.replace("九宫格的", "九宫格")
    # 去除句末标点
    text = text.rstrip('.,;!?。，；！？')
    return text


def semantic_match(pred_text, gt_text):
    """语义匹配：比较火灾判断和九宫格位置是否一致"""
    # 提取火灾判断
    pred_fire = extract_fire_decision(pred_text)
    gt_fire = extract_fire_decision(gt_text)

    if pred_fire != gt_fire:
        return False

    # 提取九宫格位置（集合比较，不考虑顺序）
    pred_positions = extract_grid_positions(pred_text)
    gt_positions = extract_grid_positions(gt_text)

    # 位置集合必须完全相同
    return pred_positions == gt_positions


def check_format_compliance(text, is_fire_gt):
    """检查格式合规性"""
    text = text.strip()

    # 基本格式检查
    if is_fire_gt:
        # 火灾图像必须包含判断语句（灵活匹配："是" + "火灾"）
        has_fire_statement = False
        if "是" in text and "火灾" in text:
            is_pos = text.find("是")
            fire_pos = text.find("火灾")
            if is_pos < fire_pos and fire_pos - is_pos < 20:
                has_fire_statement = True

        if not has_fire_statement:
            return False

        # 必须包含至少一个九宫格位置
        if not extract_grid_positions(text):
            return False
    else:
        # 非火灾图像必须明确说明（灵活匹配："不" + "火灾"）
        has_non_fire_statement = False
        if "不" in text and "火灾" in text:
            not_pos = text.find("不")
            fire_pos = text.find("火灾")
            if not_pos < fire_pos and fire_pos - not_pos < 20:
                has_non_fire_statement = True

        if not has_non_fire_statement:
            return False

        # 不应该包含九宫格位置分析
        if extract_grid_positions(text):
            return False

    return True


# ========================= 推理函数 =========================
processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)


def predict(messages, model):
    """单条数据推理"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 清理显存
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

    return output_text[0]


# 注意：calculate_loss 函数已移除，因为会导致显存不足
# 如果需要计算 loss，建议单独运行或减少 batch size


# ========================= 主验证流程 =========================
def main():
    # 1. 加载基座模型
    print(f"加载基座模型: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 2. 条件加载 LoRA 权重
    if USE_LORA:
        print(f"加载 LoRA checkpoint: {CHECKPOINT_PATH}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET_MODULES,
            inference_mode=True,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        peft_model = PeftModel.from_pretrained(model, model_id=CHECKPOINT_PATH, config=lora_config)
        print("使用 LoRA 微调模型进行验证")
    else:
        peft_model = model  # 直接使用基座模型
        print("使用基座模型（未微调）进行验证")

    # 3. 读取数据集并随机抽样
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.seed(SEED)
    samples = random.sample(dataset, min(NUM_SAMPLES, len(dataset)))
    print(f"从 {len(dataset)} 条数据中随机抽取 {len(samples)} 条进行验证")

    # 4. 初始化 SwanLab
    swanlab.init(
        project=SWANLAB_PROJECT,
        experiment_name=SWANLAB_EXPERIMENT,
        config={
            "model": MODEL_PATH,
            "use_lora": USE_LORA,
            "checkpoint": CHECKPOINT_PATH if USE_LORA else None,
            "num_samples": len(samples),
            "prompt": PROMPT,
        }
    )

    # 5. 逐条推理并计算指标
    results = []
    metrics = {
        "fire_detection_correct": 0,
        "grid_iou_sum": 0.0,
        "grid_iou_count": 0,  # 只统计火灾图像的 IoU
        "exact_match": 0,
        "format_compliance": 0,
    }

    for i, item in enumerate(samples):
        input_prompt = item["conversations"][0]["value"]
        ground_truth = item["conversations"][1]["value"]
        image_path = input_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": IMAGE_HEIGHT,
                    "resized_width": IMAGE_WIDTH,
                },
                {"type": "text", "text": PROMPT},
            ],
        }]

        # 推理
        prediction = predict(messages, peft_model)

        # 提取 ground truth 信息
        gt_is_fire = extract_fire_decision(ground_truth)
        gt_positions = extract_grid_positions(ground_truth)

        # 提取预测信息
        pred_is_fire = extract_fire_decision(prediction)
        pred_positions = extract_grid_positions(prediction)

        # 计算各项指标
        # 1. 火灾判断准确率
        fire_correct = (pred_is_fire == gt_is_fire)
        if fire_correct:
            metrics["fire_detection_correct"] += 1

        # 2. 九宫格 IoU（仅对火灾图像计算）
        grid_iou = None
        if gt_is_fire:
            grid_iou = calculate_iou(pred_positions, gt_positions)
            metrics["grid_iou_sum"] += grid_iou
            metrics["grid_iou_count"] += 1

        # 3. Exact Match (语义匹配：火灾判断 + 九宫格位置集合完全一致)
        exact_match = semantic_match(prediction, ground_truth)
        if exact_match:
            metrics["exact_match"] += 1

        # 4. 格式合规率
        format_ok = check_format_compliance(prediction, gt_is_fire)
        if format_ok:
            metrics["format_compliance"] += 1

        # 记录详细结果
        result = {
            "id": item["id"],
            "image": image_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "fire_correct": fire_correct,
            "grid_iou": grid_iou,
            "exact_match": exact_match,
            "format_ok": format_ok,
        }
        results.append(result)

        print(f"[{i+1}/{len(samples)}] {image_path}")
        print(f"  GT: {ground_truth[:80]}...")
        print(f"  Pred: {prediction[:80]}...")
        iou_str = f"{grid_iou:.3f}" if grid_iou is not None else "N/A"
        print(f"  Fire: {fire_correct} | IoU: {iou_str} | EM: {exact_match} | Format: {format_ok}")

    # 6. 汇总指标
    total = len(samples)
    final_metrics = {
        "Fire Detection Accuracy": metrics["fire_detection_correct"] / total,
        "Grid IoU (Fire only)": metrics["grid_iou_sum"] / metrics["grid_iou_count"] if metrics["grid_iou_count"] > 0 else 0.0,
        "Exact Match": metrics["exact_match"] / total,
        "Format Compliance": metrics["format_compliance"] / total,
    }

    print("\n" + "="*60)
    print("验证结果汇总:")
    print("="*60)
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.2%}")
    print("="*60)

    # 7. 保存结果到文件
    output_data = {
        "config": {
            "model": MODEL_PATH,
            "use_lora": USE_LORA,
            "checkpoint": CHECKPOINT_PATH if USE_LORA else None,
            "num_samples": total,
            "seed": SEED,
        },
        "metrics": final_metrics,
        "details": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {OUTPUT_FILE}")

    # 8. 上传到 SwanLab
    swanlab.log(final_metrics)

    # 随机选择几个样本可视化
    sample_images = []
    for result in random.sample(results, min(10, len(results))):
        caption = f"GT: {result['ground_truth'][:50]}...\nPred: {result['prediction'][:50]}..."
        sample_images.append(swanlab.Image(result["image"], caption=caption))
    swanlab.log({"Sample Predictions": sample_images})

    swanlab.finish()
    print("验证完成")


if __name__ == "__main__":
    main()
