#!/usr/bin/env python3
"""
Script to generate a text file listing all image paths in a directory.
Suitable for YOLO training on Linux systems.
使用方法：
# 基本用法（生成绝对路径）
python generate_image_list.py -i data/images -o train.txt

# 使用相对路径（推荐用于Linux）
python generate_image_list.py -i data/images -o train.txt -r

# 指定相对路径的基准目录
python generate_image_list.py -i data/images -o train.txt -r -b /path/to/yolov11
"""

import os
import argparse
from pathlib import Path


def get_image_files(input_dir, extensions=None):
    """Get all image files from the input directory."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    image_files = []
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)

    return sorted(image_files)


def generate_image_list(input_dir, output_file, use_relative_paths=False, base_dir=None):
    """
    Generate a text file with image paths.

    Args:
        input_dir: Directory containing images
        output_file: Output text file path
        use_relative_paths: If True, use relative paths instead of absolute
        base_dir: Base directory for relative paths (defaults to output file's parent)
    """
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for img_path in image_files:
            if use_relative_paths:
                if base_dir:
                    rel_path = os.path.relpath(img_path, base_dir)
                else:
                    rel_path = os.path.relpath(img_path, output_path.parent)
                # Convert Windows paths to Unix format for Linux compatibility
                path_str = rel_path.replace('\\', '/')
            else:
                path_str = str(img_path).replace('\\', '/')

            f.write(f"{path_str}\n")

    print(f"Generated {output_file} with {len(image_files)} image paths")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a text file listing all image paths for YOLO training'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing images (e.g., data/images)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output text file path (e.g., train.txt)'
    )
    parser.add_argument(
        '-r', '--relative',
        action='store_true',
        help='Use relative paths instead of absolute paths'
    )
    parser.add_argument(
        '-b', '--base-dir',
        help='Base directory for relative paths (optional)'
    )

    args = parser.parse_args()

    generate_image_list(
        args.input,
        args.output,
        use_relative_paths=args.relative,
        base_dir=args.base_dir
    )


if __name__ == '__main__':
    main()
