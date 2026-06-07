"""
Qwen2.5-VL 加载代码参考（选择A：LoRA adapter 方式）
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./model/qwen2.5-vl-fire"  # LoRA adapter 文件目录

# 加载基座模型
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 加载 LoRA adapter
model_qwen = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model_qwen.eval()

# 加载 processor
processor_qwen = AutoProcessor.from_pretrained(
    BASE_MODEL_NAME,
    min_pixels=224 * 224,
    max_pixels=1280 * 1280,
)
