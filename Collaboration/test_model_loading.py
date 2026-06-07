"""
验证模型加载脚本
在新机器上搬好模型文件后，运行此脚本检查两个模型是否能正常加载。
用法：python test_model_loading.py
"""
import os
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ============ 测试 YOLO 加载 ============
print("=" * 40)
print("测试 YOLO 模型加载...")
print("=" * 40)

YOLO_MODEL_PATH = "./model/yolo11-fire/best.onnx"

try:
    from ultralytics import YOLO
    assert os.path.exists(YOLO_MODEL_PATH), f"文件不存在: {YOLO_MODEL_PATH}"
    yolo_model = YOLO(YOLO_MODEL_PATH)
    # 打印类别信息
    print(f"YOLO 模型加载成功")
    # ONNX 模型可能不包含类名，打印检查
    names = getattr(yolo_model, 'names', None)
    if names and not all(str(v).isdigit() for v in names.values()):
        print(f"类别: {names}")
    else:
        print(f"类别: ONNX 模型未包含类名，将使用手动映射 {{0: 'fire', 1: 'smoke'}}")
except Exception as e:
    print(f"YOLO 加载失败: {e}")

print()

# ============ 测试 Qwen2.5-VL 加载 ============
print("=" * 40)
print("测试 Qwen2.5-VL 模型加载...")
print("=" * 40)

QWEN_ADAPTER_PATH = "./model/qwen2.5-vl-fire"
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    # 加载基座模型
    print(f"加载基座模型 {BASE_MODEL_NAME} ...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA adapter
    print(f"加载 LoRA adapter: {QWEN_ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, QWEN_ADAPTER_PATH)

    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME,
        min_pixels=224 * 224,
        max_pixels=1280 * 1280,
    )

    print(f"Qwen2.5-VL 模型加载成功")
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"模型精度: {next(model.parameters()).dtype}")
except Exception as e:
    print(f"Qwen 加载失败: {e}")

print()
print("验证完成。")
