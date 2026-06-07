# VLM 火灾图像理解模块

基于 Qwen2.5-VL-3B-Instruct 的视觉语言模型微调，任务为判断图像是否为火灾，并定位火焰在九宫格中的位置。采用 LoRA 微调方式，经过四轮实验迭代，最终模型以 LoRA rank=64 + 优化提示词 为最佳。

---

## 目录结构

```
qwen/
├── Qwen2.5-VL 图片自动标注工具+结果/   # Step 1: 用模型自动生成初始标注
│   ├── annotate_images.py
│   ├── my_annotations.json
│   └── requirements.txt
├── 修改json文件代码+修改的各个结果/     # Step 2: 对标注数据做后处理/修正
│   ├── add_grid_description.py
│   ├── update_user_question.py
│   └── my_annotations_updated_final_prompt.json
└── qwen2.5-vl/                          # Step 3: 正式训练/推理/验证
    ├── train_new.py
    ├── inference.py
    ├── validate_new.py
    └── validation_results_*.json
```

---

## 环境配置

**训练平台**：AutoDL 云服务器
- OS：Ubuntu 22.04
- GPU：Tesla V100-PCIE-32GB
- CUDA：≥ 12.4
- PyTorch：≥ 2.4.0

进入 `/root/autodl-tmp/` 后执行：

```bash
# 加速访问 HuggingFace（AutoDL 专用）
source /etc/network_turbo
# 或设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
# 修改模型下载位置（避免占用系统盘）
export HF_HOME=/root/autodl-tmp/huggingface/
```

安装依赖：

```bash
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils
pip install sentencepiece==0.2.0
pip install datasets==2.18.0
pip install peft==0.13.2
pip install swanlab
pip install bitsandbytes
```

---

## Step 1：自动标注数据集

用 Qwen2.5-VL-3B-Instruct 对火灾图片进行初始标注，生成微调所需的 JSON 格式数据集。

```bash
cd "Qwen2.5-VL 图片自动标注工具+结果"

# 基本用法
python annotate_images.py --image_dir /path/to/your/images

# 完整参数
python annotate_images.py \
  --image_dir /root/autodl-tmp/data/images/ \
  --output /root/autodl-tmp/data/my_annotations.json \
  --prompt "COCO Yes: " \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --device cuda
```

首次运行会自动下载模型（约 6GB）。脚本每处理 10 张自动保存一次，防止中断丢失数据。

---

## Step 2：后处理标注数据

生成初始标注后，需对 JSON 文件进行修正和增强：

```bash
cd "修改json文件代码+修改的各个结果"

# 为标注添加九宫格位置描述
python add_grid_description.py

# 统一修改 user 提问语句
python update_user_question.py
```

最终使用的标注文件为 `my_annotations_updated_final_prompt.json`，包含九宫格定位描述和完整的对话格式。

---

## Step 3：LoRA 微调训练

所有命令在 `qwen2.5-vl/` 目录下执行。

### 训练前配置

在 `qwen/qwen2.5-vl`中新建 `images`文件夹，将用于训练的图片存入。

打开 `train_new.py`，修改顶部配置区的路径：

```python
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"  # 模型路径（HuggingFace 或本地）
ANNOTATION_FILE = "my_annotations_updated_final_prompt.json"
OUTPUT_DIR = "./output/Qwen2.5-VL-3B"
```

###  执行训练

```bash
cd qwen/qwen2.5-vl
python train_new.py
```

训练过程指标通过 SwanLab 可视化，`output/` 和 `swanlog/` 为训练过程中自动生成，无需手动创建。

### 步数参考

```
每 epoch 步数 = ceil(608 / (batch_size × gradient_accumulation_steps))
             = ceil(608 / (4 × 4)) = 38 步/epoch
总步数 = 38 × 4 epoch = 152 步
```

---

## 注：关闭 QLoRA 量化方法

如不需要 4bit 量化，在 `train_new.py` 中：

1. 注释掉 `QUANTIZATION_BIT = 4` 这行
2. 将模型加载部分替换为：

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.enable_input_require_grads()
```

即去掉 `BitsAndBytesConfig`、`quantization_config` 参数和 `prepare_model_for_kbit_training` 三处。

---

## 推理

修改 `inference.py` 顶部的 checkpoint 路径后执行：

```bash
python inference.py
```

---

## 验证

修改 `validate_new.py` 顶部的 checkpoint 路径后执行：

```bash
python validate_new.py
```

各版本验证结果保存在 `qwen2.5-vl/` 下：

| 文件 | 说明 |
|---|---|
| `validation_results_baseline.json` | 未微调基线 |
| `validation_results_lora_16.json` | rank=16 结果 |
| `validation_results_lora_16_prompt.json` | rank=16+改进 prompt 结果 |
| `validation_results_lora_64.json` | rank=64+s改进 prompt 结果 |

---

最终微调权重（LoRA adapter）已整合至 `../Collaboration/model/qwen2.5-vl-fire/` 供协同推理使用。