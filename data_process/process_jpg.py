#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JPG图像处理脚本：插值放大 + 底部补零
输入：JPG图片（3通道 BGR）
处理：
1. 插值放大到 1920×1080
2. 底部补0到 1920×1536
输出：处理后的JPG图片
"""

import cv2
import numpy as np
import os
import argparse
import contextlib
import io
import multiprocessing as mp


def _process_worker(task):
    """多进程任务包装：屏蔽单图打印，返回 (文件名, 错误信息或None)"""
    processor, input_path, output_path = task
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            processor.process(input_path, output_path)
        return (os.path.basename(input_path), None)
    except Exception as e:
        return (os.path.basename(input_path), str(e))


class JpgImageProcessor:
    """JPG图像处理器：插值放大 + 底部补零"""

    INTERP_MAP = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos4': cv2.INTER_LANCZOS4,
    }

    def __init__(self, target_width: int = 1920,
                 upscale_height: int = 1080,
                 final_height: int = 1536,
                 interpolation: str = 'cubic',
                 jpg_quality: int = 95) -> None:
        """
        Args:
            target_width: 目标宽度
            upscale_height: 插值放大后的高度
            final_height: 最终高度（含补零区域）
            interpolation: 插值方法 ('nearest'/'linear'/'cubic'/'lanczos4')
            jpg_quality: JPG保存质量 (0-100)
        """
        self.target_width = target_width
        self.upscale_height = upscale_height
        self.final_height = final_height
        self.interpolation = self.INTERP_MAP[interpolation]
        self.jpg_quality = jpg_quality

    def process(self, input_path: str, output_path: str) -> None:
        """
        处理单张JPG图片：插值放大后底部补零

        Args:
            input_path: 输入JPG图片路径
            output_path: 输出JPG图片路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图片: {input_path}")

        print(f"原始图片尺寸: {img.shape[1]}×{img.shape[0]}")

        if (img.shape[1], img.shape[0]) == (self.target_width, self.upscale_height):
            img_upscaled = img
            print("尺寸已匹配，跳过插值")
        else:
            img_upscaled = cv2.resize(img, (self.target_width, self.upscale_height),
                                      interpolation=self.interpolation)
            print(f"插值放大后尺寸: {img_upscaled.shape[1]}×{img_upscaled.shape[0]}")

        pad_height = self.final_height - self.upscale_height
        if pad_height > 0:
            padding = np.zeros((pad_height, self.target_width, img_upscaled.shape[2]),
                               dtype=img_upscaled.dtype)
            img_padded = np.vstack([img_upscaled, padding])
            print(f"补零后最终尺寸: {img_padded.shape[1]}×{img_padded.shape[0]} "
                  f"(底部补零 {pad_height} 像素)")
        else:
            img_padded = img_upscaled
            print(f"无需补零，最终尺寸: {img_padded.shape[1]}×{img_padded.shape[0]}")

        cv2.imwrite(output_path, img_padded,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality])
        print(f"已保存到: {output_path}")

    def batch_process(self, input_dir: str, output_dir: str,
                      workers: int = 1) -> None:
        """
        批量处理目录下所有JPG图片

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            workers: 并行进程数 (1 表示串行)
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        jpg_files = [f for f in os.listdir(input_dir)
                     if f.lower().endswith(('.jpg', '.jpeg'))]
        if not jpg_files:
            print(f"在 {input_dir} 中未找到JPG文件")
            return

        print(f"找到 {len(jpg_files)} 个JPG文件\n")

        if workers > 1:
            print(f"使用 {workers} 个进程并行处理")
            import time
            start = time.time()
            failed = []
            with mp.Pool(workers) as pool:
                for i, (filename, err) in enumerate(pool.imap_unordered(
                        _process_worker,
                        [(self, os.path.join(input_dir, f),
                          os.path.join(output_dir, f)) for f in jpg_files]), 1):
                    if err:
                        failed.append((filename, err))
                    if i % 500 == 0 or i == len(jpg_files):
                        elapsed = time.time() - start
                        print(f"[{i}/{len(jpg_files)}] 已用时 {elapsed:.0f}s, "
                              f"失败 {len(failed)}")
            for filename, err in failed:
                print(f"  错误: {filename}: {err}")
            return

        for i, filename in enumerate(jpg_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"[{i}/{len(jpg_files)}] 处理: {filename}")
            try:
                self.process(input_path, output_path)
            except Exception as e:
                print(f"  错误: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='JPG图像插值放大与底部补零工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单张图片
  python process_jpg.py -i input.jpg -o output.jpg

  # 批量处理目录
  python process_jpg.py --input-dir ./images --output-dir ./output
        """)

    parser.add_argument('-i', '--input', type=str,
                        help='输入JPG图片路径')
    parser.add_argument('-o', '--output', type=str,
                        help='输出JPG图片路径')
    parser.add_argument('--input-dir', type=str,
                        help='输入目录（批量处理）')
    parser.add_argument('--output-dir', type=str,
                        help='输出目录（批量处理）')
    parser.add_argument('--width', type=int, default=1920,
                        help='目标宽度 (默认: 1920)')
    parser.add_argument('--upscale-height', type=int, default=1080,
                        help='插值放大后的高度 (默认: 1080)')
    parser.add_argument('--final-height', type=int, default=1536,
                        help='最终高度 (默认: 1536)')
    parser.add_argument('--interpolation', type=str, default='cubic',
                        choices=['nearest', 'linear', 'cubic', 'lanczos4'],
                        help='插值方法 (默认: cubic)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPG保存质量 0-100 (默认: 95)')
    parser.add_argument('--workers', type=int, default=1,
                        help='批量处理并行进程数 (默认: 1 串行)')

    args = parser.parse_args()

    processor = JpgImageProcessor(
        target_width=args.width,
        upscale_height=args.upscale_height,
        final_height=args.final_height,
        interpolation=args.interpolation,
        jpg_quality=args.quality)

    try:
        if args.input and args.output:
            processor.process(args.input, args.output)
        elif args.input_dir and args.output_dir:
            processor.batch_process(args.input_dir, args.output_dir,
                                    workers=args.workers)
        else:
            parser.print_help()
    except Exception as e:
        print(f"错误: {e}")
        exit(1)


if __name__ == '__main__':
    main()
