#!/usr/bin/env python3
"""
Script to generate a text file listing all image paths in a directory.
Suitable for YOLO training on Linux systems.
使用方法：
# 基本用法（生成绝对路径）
python generate_image_list.py -i data/images -o train.txt

# 使用相对路径（推荐用于Linux）
生成单个文件（不分割）：
python generate_image_list.py -i data/images -o train.txt -r

# 指定相对路径的基准目录
python generate_image_list.py -i data/images -o train.txt -r -b /path/to/yolov11

# 自动分割训练集和验证集（8:2比例）
python generate_image_list.py -i data/images --split --train-output train.txt --val-output val.txt -r
"""

import os
import argparse
import random
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


def format_path(img_path, use_relative_paths, base_dir, output_path):
    """Format image path according to settings."""
    if use_relative_paths:
        if base_dir:
            rel_path = os.path.relpath(img_path, base_dir)
        else:
            rel_path = os.path.relpath(img_path, output_path.parent)
        return rel_path.replace('\\', '/')
    else:
        return str(img_path).replace('\\', '/')


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
            path_str = format_path(img_path, use_relative_paths, base_dir, output_path)
            f.write(f"{path_str}\n")

    print(f"Generated {output_file} with {len(image_files)} image paths")


def generate_train_val_split(input_dir, train_output, val_output, train_ratio=0.8,
                             use_relative_paths=False, base_dir=None, seed=42):
    """
    Generate train.txt and val.txt with automatic split.

    Args:
        input_dir: Directory containing images
        train_output: Output file for training set
        val_output: Output file for validation set
        train_ratio: Ratio of training set (default 0.8 for 80%)
        use_relative_paths: If True, use relative paths instead of absolute
        base_dir: Base directory for relative paths
        seed: Random seed for reproducibility
    """
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(image_files)

    # Split into train and val
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Write train.txt
    train_path = Path(train_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f:
        for img_path in train_files:
            path_str = format_path(img_path, use_relative_paths, base_dir, train_path)
            f.write(f"{path_str}\n")

    # Write val.txt
    val_path = Path(val_output)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    with open(val_path, 'w', encoding='utf-8') as f:
        for img_path in val_files:
            path_str = format_path(img_path, use_relative_paths, base_dir, val_path)
            f.write(f"{path_str}\n")

    print(f"Generated {train_output} with {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"Generated {val_output} with {len(val_files)} images ({(1-train_ratio)*100:.0f}%)")
    print(f"Total: {len(image_files)} images")

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
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split images into train and val sets'
    )
    parser.add_argument(
        '--train-output',
        help='Output file for training set (required with --split)'
    )
    parser.add_argument(
        '--val-output',
        help='Output file for validation set (required with --split)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8 for 80%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    if args.split:
        if not args.train_output or not args.val_output:
            parser.error('--split requires --train-output and --val-output')
        generate_train_val_split(
            args.input,
            args.train_output,
            args.val_output,
            train_ratio=args.train_ratio,
            use_relative_paths=args.relative,
            base_dir=args.base_dir,
            seed=args.seed
        )
    else:
        if not args.output:
            parser.error('--output is required when not using --split')
        generate_image_list(
            args.input,
            args.output,
            use_relative_paths=args.relative,
            base_dir=args.base_dir
        )


if __name__ == '__main__':
    main()
