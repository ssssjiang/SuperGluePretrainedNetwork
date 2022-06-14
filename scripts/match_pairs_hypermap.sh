#!/usr/bin/env bash

INPUT_ROOT_DIR=$1
INPUT_PAIRS=$2
INPUT_IMAGE_DIR=$3
OUTPUT_DIR=$4
DATABASE=$5
TEST_TYPE=$6


if [ "${TEST_TYPE}" == "800-360-512" ];then
  echo "Start hfnet-test 800-360-512 match."
  python3 match_pairs_hypermap.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_root_dir="${INPUT_ROOT_DIR}" \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 800-360-512 match."
fi

if [ "${TEST_TYPE}" == "800-360-1024" ];then
  echo "Start hfnet-test 800-360-1024 match."
  python3 match_pairs_hypermap.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_root_dir="${INPUT_ROOT_DIR}" \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 800-360-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-1024" ];then
  echo "Start test 1040-450-1024 match."
  python3 match_pairs_hypermap.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_root_dir="${INPUT_ROOT_DIR}" \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 1040-450-1024 match."
fi

if [ "${TEST_TYPE}" == "1040-450-512" ];then
  echo "Start test 1040-450-1024 match."
  python3 match_pairs_hypermap.py --eval \
    --fast_viz \
    --viz \
    --show_keypoints \
    --input_root_dir="${INPUT_ROOT_DIR}" \
    --input_pairs="${INPUT_PAIRS}" \
    --input_dir="${INPUT_IMAGE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --database="${DATABASE}"
  echo "Finish hfnet-test 1040-450-512 match."
fi