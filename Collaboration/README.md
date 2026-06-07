# Qwen2.5-VL 图片描述训练

使用Qwen2.5-VL-3B-Instruct模型和yolo11模型，对火灾图片进行协同推理

## 一、创建统一推理环境

AutoDL选择V100-32G，cuda<=13.0，选大一点向下兼容 \
进入`/root/autodl-tmp/`打开终端 \
（如果需要将Conda环境迁移到数据盘的完整流程，请参照"D:\VLM\fire-yolo-vlm\yolo\操作.docx"文件第5页） \
创建一个新的Conda环境，同时安装两个模型所需的依赖：

```bash
# 创建新环境（Python 3.10兼容性较好）
conda create -n fire python=3.10
conda activate fire

# 安装PyTorch（根据你的CUDA版本选择命令）
# CUDA 11.8版本
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# 卸载语句
pip uninstall torch torchvision torchaudio -y
pip uninstall numpy -y && pip install "numpy<2.0.0"
```

安装完成后检验CUDA是否可用：

```bash
# 命令行运行
python -c "import torch; print('版本:', torch.__version__); print('GPU可用:', torch.cuda.is_available()); print('显卡名:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

# 输出如下即安装成功
版本: 2.4.0+cu118
GPU可用: True
显卡名: Tesla V100-PCIE-32GB
```

## 二、安装Qwen2.5-VL依赖

- AutoDL提供加速访问github/huggingface方式`source /etc/network_turbo`
- 如果网络慢，设置 `export HF_ENDPOINT=https://hf-mirror.com`
- 更改模型下载位置 `export HF_HOME=/root/autodl-tmp/huggingface/`

```bash
# 安装transformers和qwen相关依赖
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils peft

# 安装supervision（用于可视化YOLO和VLM结果）[citation:1]
pip install supervision

# 可选：如果你需要处理视频或图像增强
pip install opencv-python
```

加载Qwen2.5-VL模型：
```bash
python Qwen2.5-VL的加载代码参考.py
```

# 三、安装YOLO依赖
```bash
# YOLO通用依赖
pip install ultralytics==8.3.185
pip uninstall ultralytics -y
用训练时的版本号
```
加载YOLO模型：
```bash
python yolo11的加载代码参考.py
```

# 四、将训练好的模型迁移到推理环境
模型文件准备清单：\

模型：Qwen2.5-VL-3B-Instruct \
需要准备的文件：完整的模型文件夹（包含config.json, model.safetensors等） \
存放位置建议：./models/qwen2.5-vl-fire/  \

模型：YOLO \
需要准备的文件：权重文件（.pt或.weights）和配置文件（.yaml/.cfg） \
存放位置建议：./models/yolo-fire/  \
推荐ONNX导出：将YOLO导出为ONNX格式，使用ONNX Runtime加速

# 从 YOLO 训练机上拿的文件
只需要一个文件：训练好的权重 best.pt。

你提到用过 ONNX 导出命令，那导出后会在同目录下生成一个 best.onnx。两个都拿下来，推荐用 ONNX 版本做推理（更快、不依赖训练时的 ultralytics 版本）。

```bash
# 从 YOLO 机器下载：
runs/detect/train3/weights/best.pt
runs/detect/train3/weights/best.onnx   （如果已导出）
```
如果还没导出 ONNX，在 YOLO 机器上跑一下：
```bash
yolo task=detect mode=export model=runs/detect/train3/weights/best.pt format=onnx
```

另外还需要确认一下你的 YOLO 训练用的 data.yaml（里面定义了类别名称，比如 fire、smoke）。ultralytics 的 .pt 文件内部已经打包了类别信息，但 ONNX 没有，所以如果用 ONNX 推理需要自己知道类别映射。你把 data.yaml 也拿一份。

```bash
也拿一份：
你的 data.yaml（包含 names: [fire, smoke] 之类的类别定义）
```

# 从 Qwen 训练机上拿的文件
只搬 LoRA adapter（文件小，但推理机上需要先下载基座模型）

你在 AutoDL 上设置了 HF_HOME=/root/autodl-tmp/huggingface/ 的话，基座模型会缓存到数据盘，下次启动环境直接读本地缓存，不会再慢了。

从 Qwen 训练机上拿这些文件就够了：
checkpoint-140/（或你认为效果最好的那个 checkpoint）
├── adapter_config.json            （必须）
├── adapter_model.safetensors      （必须，~142MB）
├── tokenizer.json                 （必须）
├── tokenizer_config.json          （必须）
├── special_tokens_map.json        （有就拿）
├── preprocessor_config.json       （有就拿）
└── chat_template.json             （有就拿）
不要拿 optimizer.pt、scheduler.pt、trainer_state.json，那些是训练恢复用的。

放到新机器上 model/qwen2.5-vl-fire/ 目录下即可。

到了新机器上再合并一次，以后就不需要 peft 了：
```bash
model = PeftModel.from_pretrained(base_model, "./model/qwen2.5-vl-fire")
merged = model.merge_and_unload()
merged.save_pretrained("./model/qwen2.5-vl-fire-merged")
```

# 新机器上的目录结构
Collaboration/
└── model/
    ├── yolo11-fire/
    │   ├── best.pt              （或 best.onnx）
    │   └── data.yaml            （类别定义）
    └── qwen2.5-vl-fire/
        ├── （选择A）adapter_config.json + adapter_model.safetensors + tokenizer 文件
        └── （选择B）完整模型文件（config.json, model.safetensors, tokenizer.json 等）

# 五、验证模型加载
创建一个简单的测试脚本test_model_loading.py:
```bash
python test_model_loading.py
```

# 六、使用方法

## 如果服务器上还是没有中文字体，可以装一个：
```bash
# 下载时记得确认镜像
mkdir -p /usr/share/fonts/chinese && wget -q -O /usr/share/fonts/chinese/NotoSansCJKsc-Regular.otf https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf

# 安装完验证一下
python ziti.py
# 输出
通过 fc-list 查找中文字体...
/usr/share/fonts/chinese/NotoSansCJKsc-Regular.otf: Noto Sans CJK SC:style=Regular
```

## 单张图片推理
```bash
python collaboration.py --image /path/to/test.jpg --save-vis
```

# 批量 + 自定义输出目录
```bash
python collaboration.py --image-dir ./images/ --save-vis --output-dir ./results/vis_results_100/
```
不加 --save-vis 的话，只有终端文本输出，不会生成可视化图片。加上这个参数后，每张图推理完会调用 visualize_result() 生成一张标注

## 禁用位置一致性检查（只做双重验证，不做九宫格对比）
```bash
python collaboration.py --image /path/to/test.jpg --no-grid-check
python collaboration.py --image-dir ./images/ --save-vis --output-dir ./result/no-grid-check/ --no-grid-check
```

## 输出说明
系统会依次输出：
1. VLM 分析结果（是否火灾、火焰所在板块）
2. YOLO 使用的置信度阈值（0.2 或 0.5）
3. YOLO 检测到的目标列表（类别、置信度、边界框）
4. 位置一致性检查结果（如果启用）
5. 最终预警等级：

| 预警等级 | 含义 | VLM 判断 | YOLO 检测 |
|---------|------|---------|----------|
| ALARM | 触发警报 | 火灾 | 有火/烟 |
| CAUTION | 注意防火 | 非火灾 | 有火/烟 |
| ATTENTION | 需要人为关注 | 火灾 | 无火/烟 |
| SAFE | 安全 | 非火灾 | 无火/烟 |

# 七、测试及性能优化

# 模型验证脚本
```bash
python evaluate.py --images ./images --labels ./labels --output ./results.csv
python evaluate.py --images ./images/ --labels ./labels/ --output ./results.csv
```

建议准备几类测试图片验证四种预警级别：
- 真实火灾图片（火焰明显）
- 疑似火灾图片（红色晚霞、灯光等）
- 安全场景

性能优化建议：
- 模型量化：将Qwen2.5-VL转换为INT8或FP16精度，可减少显存占用和加速
- 异步处理：如果硬件允许，可以让VLM和YOLO并行推理
- 批处理：如果是视频流处理，可以对帧进行批处理优化