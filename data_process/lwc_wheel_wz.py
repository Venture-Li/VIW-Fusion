#!/usr/bin/env python3
"""将 scene_1–20 的 wheel0/data.csv 中 wz 取反，另存 data_wz.csv。

使用 Python 标准库，无需 ROS。原文件保留，输出存在时拒绝覆盖。
时间戳和其余列保持原文本，注释、空行及换行符保留。
"""

from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(
    "/media/nio/d1361833-68be-4f80-8dc2-2821ad2bc497/VIW-Fusion/data/raw_bags"
)


def negate_wz(text):
    output = []
    count = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            output.append(line)
            continue
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        fields = body.split(",")
        if len(fields) != 7:
            raise ValueError(f"第 {line_number} 行应有 7 列")
        value = Decimal(fields[6].strip())
        if not value.is_finite():
            raise ValueError(f"第 {line_number} 行 wz 不是有限数值")
        fields[6] = str(value.copy_negate())
        output.append(",".join(fields) + ending)
        count += 1
    if not count:
        raise ValueError("CSV 中没有轮速数据")
    return "".join(output), count


def main():
    pairs = [
        (DATA_DIR / f"scene_{i}" / "wheel0" / "data.csv",
         DATA_DIR / f"scene_{i}" / "wheel0" / "data_wz.csv")
        for i in range(1, 21)
    ]
    # 先检查全部路径，避免漏掉序列或覆盖先前结果。
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有输出：{target}")

    for source, target in pairs:
        with source.open("r", encoding="utf-8", newline="") as stream:
            result, count = negate_wz(stream.read())
        with target.open("x", encoding="utf-8", newline="") as stream:
            stream.write(result)
        print(f"{source.parent.parent.name}: {count} 条 → {target}")
    print("完成：20 个序列的 wz 已取反，原 data.csv 均保留。")


if __name__ == "__main__":
    main()
