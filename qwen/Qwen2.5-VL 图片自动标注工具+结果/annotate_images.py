#!/usr/bin/env python3
"""
使用Qwen2.5-VL-3B-Instruct自动标注图片生成JSON数据集
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

try:
    from PIL import Image
except ImportError:
    print("警告: 未安装Pillow，无法调整图片大小")
    Image = None

# Pillow 10+ 兼容性：优先使用新的 Resampling 枚举
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

class ImageAnnotator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        """初始化标注器"""
        self.device = device
        print(f"加载模型: {model_name}")

        # 加载模型和处理器 - 使用8bit量化节省显存
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto",
            ignore_mismatched_sizes=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        print("模型加载完成")

    def annotate_single_image(self, image_path: str, prompt: str = "请先判断是否为火灾图像，再详细描述这张图片的内容。", max_size: int = 1024) -> str:
        """标注单张图片"""
        # 加载并调整图片大小以节省显存
        from PIL import Image
        img = Image.open(image_path)

        # 如果图片过大，按比例缩小
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, LANCZOS)  # 使用全局定义的LANCZOS

            # 保存临时缩小的图片
            temp_path = f"/tmp/temp_resized_{Path(image_path).name}"
            img.save(temp_path)
            image_path = temp_path

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # 生成描述
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            caption = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return caption.strip()
        finally:
            # 清理显存
            del inputs
            if 'generated_ids' in locals():
                del generated_ids
            torch.cuda.empty_cache()

    def annotate_batch(
        self,
        image_dir: str,
        output_file: str,
        prompt: str = "COCO Yes: ",
        max_size: int = 1024,
        image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ) -> List[Dict]:
        """批量标注图片"""
        image_dir = Path(image_dir)

        if not image_dir.exists():
            raise ValueError(f"图片目录不存在: {image_dir}")

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)
        print(f"找到 {len(image_files)} 张图片")

        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中没有找到图片文件")

        # 标注数据集
        dataset = []

        for idx, image_path in enumerate(tqdm(image_files, desc="标注进度")):
            try:
                # 生成描述（传入max_size参数）
                caption = self.annotate_single_image(str(image_path), prompt, max_size)

                # 构建数据条目（Qwen2.5-VL微调格式）
                entry = {
                    "id": f"identity_{idx + 1}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": f"{prompt}<|vision_start|>{image_path.name}<|vision_end|>"
                        },
                        {
                            "from": "assistant",
                            "value": caption
                        }
                    ]
                }

                dataset.append(entry)

                # 每处理10张图片保存一次（防止意外中断丢失数据）
                if (idx + 1) % 10 == 0:
                    self._save_dataset(dataset, output_file)

            except Exception as e:
                print(f"\n处理 {image_path.name} 时出错: {e}")
                continue

        # 最终保存
        self._save_dataset(dataset, output_file)
        print(f"\n标注完成！共标注 {len(dataset)} 张图片")
        print(f"数据已保存到: {output_file}")

        return dataset

    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """保存数据集"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="使用Qwen2.5-VL自动标注图片")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--output", type=str, default="annotations.json", help="输出JSON文件路径")
    parser.add_argument("--prompt", type=str, default="请先判断是否为火灾图像，再详细描述这张图片的内容。", help="提示词")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="模型名称")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    parser.add_argument("--max_size", type=int, default=1024, help="图片最大尺寸（像素），用于节省显存")

    args = parser.parse_args()

    # 自动选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"使用设备: {device}")
    print(f"图片目录: {args.image_dir}")
    print(f"输出文件: {args.output}")
    print(f"图片最大尺寸: {args.max_size}px")

    # 创建标注器
    annotator = ImageAnnotator(model_name=args.model, device=device)

    # 批量标注
    annotator.annotate_batch(
        image_dir=args.image_dir,
        output_file=args.output,
        prompt=args.prompt,
        max_size=args.max_size
    )


if __name__ == "__main__":
    main()
