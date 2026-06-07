# Qwen2.5-VL 图片自动标注工具

使用Qwen2.5-VL-3B-Instruct模型自动标注图片，生成符合微调格式的JSON数据集。

## 安装依赖

AutoDL选择V100-32G，cuda>=12.4，PyTorch>=2.4.0

进入`/root/autodl-tmp/`打开终端

- AutoDL提供加速访问github/huggingface方式`source /etc/network_turbo`
- 如果网络慢，设置 `export HF_ENDPOINT=https://hf-mirror.com`
- 更改模型下载位置 `export HF_HOME=/root/autodl-tmp/huggingface/`

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
```
参照requirements.txt确保都下完整

## 使用方法

### 基本用法

```bash
python annotate_images.py --image_dir /path/to/your/images
```

### 完整参数

```bash
python annotate_images.py \
  --image_dir /path/to/your/images \
  --output annotations.json \
  --prompt "COCO Yes: " \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --device cuda
```

### 参数说明

- `--image_dir`: **必需**，图片目录路径
- `--output`: 输出JSON文件路径，默认为 `annotations.json`
- `--prompt`: 提示词，默认为 `COCO Yes: `
- `--model`: 模型名称，默认为 `Qwen/Qwen2.5-VL-3B-Instruct`
- `--device`: 设备选择 (cuda/cpu/auto)，默认为 auto

## 输出格式

生成的JSON文件格式符合Qwen2.5-VL微调要求：

```json
[
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "COCO Yes: <|vision_start|>image_001.jpg<|vision_end|>"
      },
      {
        "from": "assistant",
        "value": "图片描述内容..."
      }
    ]
  }
]
```

## 示例

假设你的图片在 `/root/autodl-tmp/data/images/` 目录下：

```bash
python annotate_images.py --image_dir /root/autodl-tmp/data/images/ --output /root/autodl-tmp/data/my_annotations.json
python annotate_images.py --image_dir /root/autodl-tmp/test/ --output /root/autodl-tmp/test/my_annotations.json
```

## 注意事项

1. 支持的图片格式：jpg, jpeg, png, bmp, webp
2. 脚本会每处理10张图片自动保存一次，防止意外中断丢失数据
3. 首次运行会自动下载模型（约6GB）
4. 建议使用GPU加速，500张图片预计需要30-60分钟

## 步数计算

```bash
每 epoch 步数 = ceil(训练样本数 / (batch_size × gradient_accumulation_steps))
            ≈ ceil(608 / (4 × 4))
            = ceil(608 / 16)
            = 38 步/epoch

总步数 = 38 × 5 epoch = 190 步
```