#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并指定目录下所有 txt 文件内容（按文件名排序），输出到合并文件。
用法: python3 merge_txt.py /path/to/speed_dir [输出文件名]
"""

import os
import sys
import glob
from pathlib import Path

def merge_txt_files(directory, output_file):
    # 获取所有 txt 文件，按文件名排序
    txt_files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    
    if not txt_files:
        print("错误: 未找到任何 .txt 文件")
        return

    print(f"找到 {len(txt_files)} 个 txt 文件")
    
    with open(output_file, 'w') as out_f:
        for file_path in txt_files:
            print(f"  合并: {os.path.basename(file_path)}")
            with open(file_path, 'r') as in_f:
                # 读取全部内容写入（不添加多余空行）
                content = in_f.read()
                if content:
                    out_f.write(content)
                    # 确保每个文件内容末尾有换行（如果原文件没有）
                    if not content.endswith('\n'):
                        out_f.write('\n')
    
    print(f"合并完成，输出文件: {output_file}")

if __name__ == "__main__":
    # 默认目录（可修改为您的实际路径）
    default_dir = "/media/nio/13357f85-3d3c-4608-9b8b-91b03de4c3bf1/unpacked/10.18/车采数据/轮速/1/"
    default_output = os.path.join(default_dir, "combined.txt")
    
    # 命令行参数解析
    if len(sys.argv) >= 2:
        dir_path = sys.argv[1]
    else:
        dir_path = default_dir
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = default_output
    
    merge_txt_files(dir_path, out_path)