#!/usr/bin/env bash
# Usage: DATA_ROOT_BASE=/path/to/data bash eval/batch_eval_svc.sh --calibration /path/to/calibration.json
set -euo pipefail
VIW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$VIW_DIR/output}"
CONFIG="${CONFIG:-$VIW_DIR/config/realsense_d455i/realsense_stereo_imu_config_d455.yaml}"
: "${DATA_ROOT_BASE:?Set DATA_ROOT_BASE to the directory containing scene_N data folders}"
mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/summary_svc.csv"
echo 'scene,status,result_json' > "$SUMMARY"
failed=0
for scene in "$OUTPUT_DIR"/scene_*; do
    [[ -d "$scene" ]] || continue
    name="$(basename "$scene")"
    [[ -s "$scene/log/pose_output.txt" && -s "$scene/log/points_output.txt" ]] || {
        echo "$name,SKIP," >> "$SUMMARY"; continue;
    }
    # Remove only the previous result marker, so a failed run cannot appear successful.
    rm -f "$scene/depth_eval_result.json"
    if python3 "$VIW_DIR/eval/eval_depth_svc.py" --scene-dir "$scene" \
        --data-root "$DATA_ROOT_BASE/$name" --config "$CONFIG" "$@" \
        > "$scene/depth_eval.log" 2>&1; then
        echo "$name,PASS,$scene/depth_eval_result.json" >> "$SUMMARY"
    else
        echo "$name,FAIL," >> "$SUMMARY"
        cat "$scene/depth_eval.log" >&2
        failed=1
    fi
done
cat "$SUMMARY"
exit "$failed"
