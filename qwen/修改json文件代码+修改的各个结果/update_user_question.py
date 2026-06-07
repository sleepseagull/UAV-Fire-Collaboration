import json

# ===== 配置区域 =====
INPUT_FILE = "my_annotations_updated_final.json"
OUTPUT_FILE = "my_annotations_updated_final_prompt.json"
# 使用方式 - 终端输入：cd /d "d:\VLM\fire-yolo-vlm\txt-box" && python update_user_question.py
# ===================

# 原始问题和新问题
OLD_QUESTION = "请先判断是否为火灾图像，再说明火焰占据图片九宫格的哪几个版块。"
NEW_QUESTION = "请先判断是否为火灾图像。如果是，请说明火焰占据图片九宫格的哪几个版块；若不是，则直接回答不是火灾图像即可，无需分析版块。"

def update_user_questions():
    """
    批量更新 JSON 文件中用户的提问内容
    """
    # 读取 JSON 文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_count = 0

    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 1:
            continue

        # 获取用户消息
        user_msg = conversations[0].get("value", "")

        # 替换问题文本
        if OLD_QUESTION in user_msg:
            new_msg = user_msg.replace(OLD_QUESTION, NEW_QUESTION)
            conversations[0]["value"] = new_msg
            updated_count += 1

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！")
    print(f"- 更新条数: {updated_count} 条")
    print(f"- 输出文件: {OUTPUT_FILE}")

def remove_detailed_description():
    """
    删除九宫格板块描述后面的详细描述部分
    保留格式：是的，这是一张火灾图像。\n\n火焰主要占据了图片九宫格的XXX板块。
    """
    # 读取 JSON 文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_count = 0

    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        # 获取助手回复
        assistant_msg = conversations[1].get("value", "")

        # 查找九宫格描述的结束位置
        if "火焰主要占据了图片九宫格的" in assistant_msg and "板块。" in assistant_msg:
            # 找到"板块。"后的第一个"\n\n"位置
            grid_start = assistant_msg.find("火焰主要占据了图片九宫格的")
            block_end = assistant_msg.find("板块。", grid_start)

            if block_end != -1:
                # 保留到"板块。"为止，删除后面的内容
                new_msg = assistant_msg[:block_end + len("板块。")]
                conversations[1]["value"] = new_msg
                updated_count += 1

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"删除详细描述完成！")
    print(f"- 处理条数: {updated_count} 条")
    print(f"- 输出文件: {OUTPUT_FILE}")

def update_image_paths(image_prefix="images/"):
    """
    更新图片路径，从单纯的文件名改为完整的相对路径
    例如：00000.jpg -> images/00000.jpg
    验证路径是否可以读取：cd /d "d:\VLM\fire-yolo-vlm\txt-box" && python -c "import os; print(os.path.exists('images/00000.jpg'))"

    参数：
        image_prefix: 图片路径前缀，默认为 "images/"
    """
    # 读取 JSON 文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_count = 0

    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 1:
            continue

        # 获取用户消息
        user_msg = conversations[0].get("value", "")

        # 查找 <|vision_start|> 和 <|vision_end|> 标签
        start_tag = "<|vision_start|>"
        end_tag = "<|vision_end|>"

        start_idx = user_msg.find(start_tag)
        end_idx = user_msg.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            # 提取文件名
            filename = user_msg[start_idx + len(start_tag):end_idx]

            # 检查是否已经包含路径前缀
            if not filename.startswith(image_prefix):
                # 添加路径前缀
                new_filename = image_prefix + filename
                new_msg = user_msg[:start_idx + len(start_tag)] + new_filename + user_msg[end_idx:]
                conversations[0]["value"] = new_msg
                updated_count += 1

    # 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"更新图片路径完成！")
    print(f"- 处理条数: {updated_count} 条")
    print(f"- 路径前缀: {image_prefix}")
    print(f"- 输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    update_user_questions()  # 更新用户提问
    # remove_detailed_description()  # 删除详细描述
    # update_image_paths("images/")  # 更新图片路径，可修改路径前缀
