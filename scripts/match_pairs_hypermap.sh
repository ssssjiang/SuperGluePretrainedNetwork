#!/usr/bin/env bash

source /persist/Hierarchical-Localization/venv/bin/activate

INPUT_ROOT_DIR=$1
INPUT_PAIRS=$2
INPUT_IMAGE_DIR=$3
DATABASE=$4
OUTPUT_DIR=$5


echo "Start hypermap-superglue match."
python3 match_pairs_hypermap.py \
  --cache --viz --fast_viz --show_keypoints \
  --input_root_dir="${INPUT_ROOT_DIR}" \
  --input_pairs="${INPUT_PAIRS}" \
  --input_dir="${INPUT_IMAGE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --database="${DATABASE}" \
  --sinkhorn_iterations=50 \
  --match_threshold=0.2
#  --crop_size -1
# --loransac
# --overwrite
echo "Finish hypermap-superglue match."
