#!/usr/bin/env bash

INPUT_PAIRS=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
DATABASE=$4
TEST_TYPE=$5


if [ "${TEST_TYPE}" == "800-360-512" ];then
  echo "Start hfnet-test 800-360-512 match."
  python3 match_pairs_hfnet_mower.py --eval \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 800-360-512 match."
fi

if [ "${TEST_TYPE}" == "800-360-1024" ];then
  echo "Start hfnet-test 800-360-1024 match."
  python3 match_pairs_hfnet_mower.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 800-360-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-1024" ];then
  echo "Start test 1040-450-1024 match."
  python3 match_pairs_hfnet_mower.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 1040-450-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-512" ];then
  echo "Start test 1040-450-1024 match."
  python3 match_pairs_hfnet_mower.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 1040-450-512 match."
fi