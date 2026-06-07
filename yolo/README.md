# YOLO 火灾烟雾检测模块

基于 YOLOv11 的火灾/烟雾目标检测，在标准 YOLOv11s 基础上引入多种注意力机制并进行消融实验，最终选用 EMA 注意力模块作为最优方案。

---

## 环境配置

**训练平台**：AutoDL 云服务器
- OS：Ubuntu 22.04
- GPU：Tesla V100-PCIE-32GB
- PyTorch：2.0.1+cu118
- Python：3.10

### 基础训练环境（yolov11）

```bash
conda create -n v11 python=3.10
conda activate v11
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r yolov11/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 注意力机制训练环境（yolov11-attention）

```bash
conda create -n a11 python=3.10
conda activate a11
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r yolov11-attention/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：安装完成后用 `conda list` 检查，**不能有 `ultralytics` 这个库**，否则会和本地修改的代码冲突。

然后将项目以可编辑模式安装：

```bash
cd yolov11-attention
pip install -e .
```

验证 GPU 可用：

```bash
python -c "import torch; print('版本:', torch.__version__); \
    print('GPU可用:', torch.cuda.is_available()); \
    print('显卡名:', torch.cuda.get_device_name(0))"
```

---

## 数据集准备

本项目使用公开火灾/烟雾数据集，需自行下载后按以下结构组织：

```
数据集根目录/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

生成索引文件（将图片路径写入 txt）：

```bash
python yolov11/generate_image_list.py
```

生成的 `train.txt` / `val.txt` / `test.txt` 放入 `fire/` 目录。

修改数据集配置文件中的路径（`ultralytics/cfg/datasets/A-fire-smoke.yaml`）：

```yaml
path: /你的数据集根目录/
train: train.txt
val: val.txt
test: test.txt
names:
  0: fire
  1: smoke
```

---

## 预训练权重下载

从 Ultralytics 官方下载，放入对应的 `weights/` 目录：

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt \
    -P yolov11-attention/weights/
```

---

## 训练流程

### 第一轮：基线（YOLOv11s，仅 fire 类）

在 `yolov11/` 目录下执行：

```bash
yolo task=detect mode=train \
    model=weights/yolo11s.pt \
    data=ultralytics/cfg/datasets/A-fire.yaml \
    batch=48 epochs=100 imgsz=640 workers=0 device=0
```

验证：

```bash
yolo task=detect mode=val \
    model=runs/detect/train/weights/best.pt \
    data=ultralytics/cfg/datasets/A-fire.yaml \
    device=0 plots=True split=test
```

### 第二轮：扩展类别（fire + smoke）

```bash
yolo task=detect mode=train \
    model=weights/yolo11s.pt \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    batch=48 epochs=100 imgsz=640 workers=0 device=0
```

如需从中断处继续训练：

```bash
yolo task=detect mode=train \
    model=runs/detect/train2/weights/last.pt \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    batch=32 epochs=100 imgsz=640 workers=4 \
    resume=True device=0 amp=True
```

验证：

```bash
yolo task=detect mode=val \
    model=runs/detect/train2/weights/best.pt \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    device=0 plots=True split=test
```

---

## 注意力机制集成（yolov11-attention）

以下步骤说明如何将自定义注意力模块接入 YOLOv11，所有命令在 `yolov11-attention/` 目录下执行。

### 代码修改步骤

1. 在 `ultralytics/nn/` 下创建 `Attmodules/` 文件夹，将注意力模块 `.py` 文件放入
2. 修改 `ultralytics/nn/tasks.py`，共 3 处：导入模块、注册到模型解析、添加到 `__all__`
3. 复制 `ultralytics/cfg/models/11/yolo11.yaml`，在 `ultralytics/cfg/models/11/Att_yaml/` 下新建对应模块的 yaml（如 `yolo11s-EMA.yaml`），在 yaml 中插入注意力模块节点

> 模型 yaml 文件名**必须含有 `s`**（如 `yolo11s-EMA.yaml`），否则预训练权重尺寸不匹配。

### 第三轮：EMA 注意力（最终选用）

```bash
yolo detect train \
    model=ultralytics/cfg/models/11/Att_yaml/yolo11s-EMA.yaml \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    batch=48 epochs=100 imgsz=640 workers=4 \
    device=0 pretrained=weights/yolo11s.pt cache=True amp=True
```

验证：

```bash
yolo task=detect mode=val \
    model=runs/detect/train3/weights/best.pt \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    device=0 plots=True split=test
```

推理：

```bash
yolo task=detect mode=predict \
    model=runs/detect/train3/weights/best.pt \
    source=test-picture device=0
```

导出 ONNX：

```bash
yolo task=detect mode=export \
    model=runs/detect/train3/weights/best.pt \
    format=onnx
```

### 第四轮：SCSA 注意力（对比实验）

```bash
yolo detect train \
    model=ultralytics/cfg/models/11/Att_yaml/yolo11s-SCSA.yaml \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    batch=48 epochs=100 imgsz=640 workers=4 \
    device=0 pretrained=weights/yolo11s.pt cache=True amp=True
```

验证：

```bash
yolo task=detect mode=val \
    model=runs/detect/train4/weights/best.pt \
    data=ultralytics/cfg/datasets/A-fire-smoke.yaml \
    device=0 plots=True split=test
```

> **已知问题**：SCSA 模块使用了全局平均池化，导出 ONNX 格式时存在不兼容问题，最终未采用该方案。

---

## 注意力模块一览

所有模块实现位于 `Attmodules/`：

| 模块 | 文件 | 类型 |
|---|---|---|
| EMA | `EMA.py` | 空间+通道（最终选用） |
| SCSA | `SCSA.py` | 空间+通道（ONNX不兼容） |
| CBAM | `CBAM.py` | 通道+空间串联 |
| SE | `SE.py` | 通道注意力 |
| CA | `CA.py` | 坐标注意力 |
| ECA | `ECA.py` | 轻量通道注意力 |
| SimAM | `SimAM.py` | 无参数注意力 |
| GAM | `GAM.py` | 全局注意力 |

对应的模型 yaml 配置均在 `yolov11-attention/ultralytics/cfg/models/11/Att_yaml/` 下。

---

最终训练好的 EMA 模型权重已整合至 `../Collaboration/model/yolo11-fire/` 供协同推理使用。
