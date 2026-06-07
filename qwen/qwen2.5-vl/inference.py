"""
Qwen2.5-VL LoRA 推理脚本
加载训练好的 LoRA 权重，对测试数据进行推理并上传结果到 SwanLab
"""
import json
import torch
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab

# ========================= 统一配置区 =========================
# 路径配置
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"           # 基座模型路径
CHECKPOINT_PATH = "./output/checkpoint-140"  # LoRA checkpoint 路径
TEST_DATA_FILE = "data_vl_test.json"                   # 测试数据文件

# 推理配置
MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 20                                    # 最少生成token数，防止模型只输出几个字
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
SWANLAB_EXPERIMENT = "4fire-yolo-grid-inference"

# ========================= 推理函数 =========================
processor = AutoProcessor.from_pretrained(MODEL_PATH)


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
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MIN_NEW_TOKENS)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ========================= 主推理流程 =========================
def main():
    # 1. 加载基座模型
    print(f"加载基座模型: {MODEL_PATH}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 2. 加载 LoRA 权重
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

    # 3. 读取测试数据
    with open(TEST_DATA_FILE, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)
    print(f"测试数据: {len(test_dataset)} 条")

    # 4. 初始化 SwanLab
    swanlab.init(project=SWANLAB_PROJECT, experiment_name=SWANLAB_EXPERIMENT)

    # 5. 逐条推理
    test_image_list = []
    for i, item in enumerate(test_dataset):
        input_prompt = item["conversations"][0]["value"]
        image_path = input_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }]

        response = predict(messages, peft_model)
        print(f"[{i+1}/{len(test_dataset)}] {image_path} -> {response}")

        test_image_list.append(swanlab.Image(image_path, caption=response))

    swanlab.log({"Prediction": test_image_list})
    swanlab.finish()
    print("推理完成")


if __name__ == "__main__":
    main()
