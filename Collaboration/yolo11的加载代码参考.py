"""
YOLO11 加载代码参考
"""
import os
import torch
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