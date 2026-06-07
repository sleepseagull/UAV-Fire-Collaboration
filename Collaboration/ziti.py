import os
import subprocess

# 1. 用 find 命令搜索系统中所有字体文件
print("=" * 50)
print("搜索系统中的字体文件 (.ttf / .ttc / .otf)...")
print("=" * 50)

result = subprocess.run(
    ["find", "/", "-name", "*.ttf", "-o", "-name", "*.ttc", "-o", "-name", "*.otf"],
    capture_output=True, text=True, timeout=30
)

fonts = [f for f in result.stdout.strip().split("\n") if f]

if fonts:
    print(f"共找到 {len(fonts)} 个字体文件:\n")
    for f in fonts:
        print(f"  {f}")
else:
    print("未找到任何字体文件。")

# 2. 额外检查 fc-list 中的中文字体
print("\n" + "=" * 50)
print("通过 fc-list 查找中文字体...")
print("=" * 50)

try:
    result2 = subprocess.run(
        ["fc-list", ":lang=zh"],
        capture_output=True, text=True, timeout=10
    )
    if result2.stdout.strip():
        print(result2.stdout.strip())
    else:
        print("fc-list 未找到中文字体。")
except FileNotFoundError:
    print("fc-list 命令不可用，跳过。")
