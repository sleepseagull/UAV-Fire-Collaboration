import os
from pathlib import Path

# 设置路径
images_dir = Path("images")
labels_dir = Path("labels")

# 确保labels目录存在
labels_dir.mkdir(exist_ok=True)

# 获取所有图片文件
image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

# 为每个图片生成标签文件
for image_file in image_files:
    # 获取不带扩展名的文件名
    image_name = image_file.stem

    # 判断是否为nofire1-10
    if image_name in [f"nofire{i}" for i in range(1, 11)]:
        label = "safe"
    else:
        label = "unsafe"

    # 创建对应的txt文件
    label_file = labels_dir / f"{image_name}.txt"
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write(label)

    print(f"{image_name}: {label}")

print(f"\n完成！共生成 {len(image_files)} 个标签文件")
