import json
import os
from pathlib import Path

# ===== 配置区域 =====
IMAGES_DIR = "images"
LABELS_DIR = "labels"
JSON_FILE = "my_annotations.json"
OUTPUT_FILE = "my_annotations_with_box.json"
# 使用方式 - 终端输入：cd /d "d:\VLM\fire-yolo-vlm\txt-box" && python add_grid_description.py
# ===================

# 九宫格位置名称（按顺序）
GRID_NAMES = [
    "左上", "中上", "右上",
    "左中", "中", "右中",
    "左下", "中下", "右下"
]

def get_grid_coverage(label_path):
    """
    读取 YOLO 标注文件，计算所有边界框覆盖的九宫格位置
    返回：覆盖的格子索引列表（0-8）
    """
    if not os.path.exists(label_path):
        return []

    covered_grids = set()

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            # YOLO 格式：class x_center y_center width height（归一化）
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 计算边界框的四个角
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            # 检查与九宫格的交集
            for row in range(3):
                for col in range(3):
                    # 九宫格边界
                    grid_x_min = col / 3
                    grid_x_max = (col + 1) / 3
                    grid_y_min = row / 3
                    grid_y_max = (row + 1) / 3

                    # 检查是否有交集
                    if (x_min < grid_x_max and x_max > grid_x_min and
                        y_min < grid_y_max and y_max > grid_y_min):
                        grid_idx = row * 3 + col
                        covered_grids.add(grid_idx)

    return sorted(covered_grids)

def extract_image_filename(vision_tag):
    """
    从 <|vision_start|>filename<|vision_end|> 中提取文件名
    """
    start_tag = "<|vision_start|>"
    end_tag = "<|vision_end|>"

    start_idx = vision_tag.find(start_tag)
    end_idx = vision_tag.find(end_tag)

    if start_idx != -1 and end_idx != -1:
        filename = vision_tag[start_idx + len(start_tag):end_idx]
        return filename
    return None

def process_annotations():
    """
    处理标注文件，添加九宫格描述
    """
    # 读取 JSON 文件
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_count = 0
    skipped_count = 0

    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        user_msg = conversations[0].get("value", "")
        assistant_msg = conversations[1].get("value", "")

        # 提取图片文件名
        image_filename = extract_image_filename(user_msg)
        if not image_filename:
            skipped_count += 1
            continue

        # 构建标注文件路径
        label_filename = Path(image_filename).stem + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        # 获取九宫格覆盖
        covered_grids = get_grid_coverage(label_path)

        if not covered_grids:
            skipped_count += 1
            continue

        # 生成描述
        grid_names = [GRID_NAMES[i] for i in covered_grids]
        grid_desc = "、".join(grid_names)
        description = f"火焰主要占据了图片九宫格的{grid_desc}板块。"

        # 在 "是的，这是一张火灾图像。\n\n" 后添加描述
        if "是的，这是一张火灾图像。" in assistant_msg:
            # 替换第一次出现的位置
            assistant_msg = assistant_msg.replace(
                "是的，这是一张火灾图像。\n\n",
                f"是的，这是一张火灾图像。\n\n{description}\n\n",
                1
            )
            conversations[1]["value"] = assistant_msg
            processed_count += 1

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！")
    print(f"- 成功处理: {processed_count} 条")
    print(f"- 跳过: {skipped_count} 条")
    print(f"- 输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_annotations()
