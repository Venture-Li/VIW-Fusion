#!/bin/bash
# 清除 output 目录：保留 scene_N/log 空文件夹，删除其余所有文件
find /home/nio/vscode/python_project_backup/VIW-Fusion/output/scene_* -maxdepth 1 -not -name log -not -type d -delete
for d in /home/nio/vscode/python_project_backup/VIW-Fusion/output/scene_*/log/; do
    rm -f "$d"/*
done
echo "Cleaned."
