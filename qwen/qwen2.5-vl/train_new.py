"""
Qwen2.5-VL LoRA 微调训练脚本
火灾图像描述 + 九宫格定位任务
"""
import json
import random
import torch
from datetime import datetime
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from swanlab.integration.transformers import SwanLabCallback
import swanlab

# ========================= 统一配置区 =========================
# 路径配置
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"       # 模型路径（HuggingFace或本地路径）
ANNOTATION_FILE = "my_annotations_updated_final_prompt.json"  # 标注文件路径
OUTPUT_DIR = "./output/Qwen2.5-VL-3B"              # 输出目录

# 数据配置
VAL_RATIO = 0.1                                     # 验证集比例
MAX_LENGTH = 8192                                    # 最大token长度
IMAGE_HEIGHT = 280                                   # 图片缩放高度
IMAGE_WIDTH = 280                                    # 图片缩放宽度
PROMPT = "请先判断是否为火灾图像。如果是，请说明火焰占据图片九宫格的哪几个版块；若不是，则直接回答不是火灾图像即可，无需分析版块。"

# LoRA 配置
LORA_RANK = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 训练超参数
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
EVAL_STEPS = 10                                      # 验证间隔（与save_steps对齐）
SAVE_STEPS = 10
LR_SCHEDULER_TYPE = "cosine"                        # cosine 退火，比 linear 更平滑
# QUANTIZATION_BIT = 4                                 # QLoRA: 4bit NF4 量化，比 8bit 更省显存且避免兼容问题
SEED = 42
# SwanLab 配置
SWANLAB_PROJECT = "Fire-Detection-VL"
SWANLAB_EXPERIMENT = "4fire-yolo-grid-detection"

# ========================= 数据预处理 =========================
# 全局加载 tokenizer 和 processor（process_func 中需要用到）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH)


def process_func(example):
    """将单条数据转换为模型输入格式"""
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": file_path,
                    "resized_height": IMAGE_HEIGHT,
                    "resized_width": IMAGE_WIDTH,
                },
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}

    response = tokenizer(output_content, add_special_tokens=False)

    # 关键：response 末尾必须加 <|im_end|> 结束标记，否则模型学不会何时停止生成
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    input_ids = inputs["input_ids"][0] + response["input_ids"] + im_end_ids
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1] * len(im_end_ids)
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + im_end_ids

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"][0]),
    }

# ========================= 主训练流程 =========================
def main():
    # 0. 生成带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_exp_{timestamp}"
    print(f"本次训练输出目录: {output_dir}")

    # 1. 加载模型（QLoRA: 4bit NF4 量化）
    print(f"加载模型: {MODEL_PATH}")
    # quantization_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_compute_dtype=torch.bfloat16,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_use_double_quant=True,
    # )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    #    quantization_config=quantization_config,
        trust_remote_code=True,
    )
    # model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # 2. 加载并拆分数据（打乱后再切分，确保火灾/非火灾样本均匀分布）
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"读取到 {len(data)} 条数据")

    random.seed(SEED)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - VAL_RATIO))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

    # map 后移除原始的非 tensor 列（id, conversations），避免 DataCollator 报错
    keep_columns = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    train_dataset = Dataset.from_list(train_data).map(process_func).remove_columns(
        [c for c in Dataset.from_list(train_data).column_names if c not in keep_columns]
    )
    val_dataset = Dataset.from_list(val_data).map(process_func).remove_columns(
        [c for c in Dataset.from_list(val_data).column_names if c not in keep_columns]
    )

    # 3. 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # 4. 训练参数
    #    bf16=True: 训练过程中使用 bfloat16 混合精度，减少显存占用并加速计算
    #    模型本身已经以 bfloat16 加载，这里让优化器的前向/反向传播也用 bf16
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,          # cosine 退火
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,                        # 混合精度训练
        gradient_checkpointing=True,      # 用计算换显存
        remove_unused_columns=False,      # 保留 pixel_values 等多模态列
        report_to="none",                 # 日志由 SwanLab 接管
        seed=SEED,
    #    save_total_limit=3,               # 最多保留3个checkpoint，节省磁盘
    )
    # 5. SwanLab 回调（自动记录 train_loss 和 eval_loss）
    swanlab_callback = SwanLabCallback(
        project=SWANLAB_PROJECT,
        experiment_name=SWANLAB_EXPERIMENT,
        config={
            "model": MODEL_PATH,
            "annotation_file": ANNOTATION_FILE,
            "prompt": PROMPT,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "image_size": f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}",
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "lr_scheduler_type": LR_SCHEDULER_TYPE,
    #        "quantization_bit": QUANTIZATION_BIT,
            "bf16": True,
        },
    )

    # 6. 启动训练
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # 7. 保存最佳模型
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"训练完成，最佳checkpoint: {best_ckpt}")
    print(f"最佳eval_loss: {trainer.state.best_metric:.4f}")

    swanlab.finish()


if __name__ == "__main__":
    main()
