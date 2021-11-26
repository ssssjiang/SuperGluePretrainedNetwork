#!/usr/bin/env bash

INPUT_PAIRS=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
TEST_TYPE=$4


if [ "${TEST_TYPE}" == "800-360-512" ];then
  echo "Start test 800-360-512 match."
  python3 match_pairs_mower.py --eval \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --max_keypoints=512 \
    --keypoint_threshold=0.005 \
    --crop_size 240 0 800 360
  echo "Finish test 800-360-512 match."
fi

if [ "${TEST_TYPE}" == "800-360-1024" ];then
  echo "Start test 800-360-1024 match."
  python3 match_pairs_mower.py --eval \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --max_keypoints=1024 \
    --keypoint_threshold=0.003 \
    --crop_size 240 0 800 360
  echo "Finish test 800-360-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-1024" ];then
  echo "Start test 1040-450-1024 match."
  python3 match_pairs_mower.py --eval \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --max_keypoints=1024 \
    --keypoint_threshold=0.003 \
    --crop_size 120 0 1040 450
  echo "Finish test 1040-450-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-512" ];then
  echo "Start test 1040-450-512 match."
  python3 match_pairs_mower.py --eval \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --max_keypoints=512 \
    --keypoint_threshold=0.005 \
    --crop_size 120 0 1040 450
  echo "Finish test 1040-450-512 match."
fi