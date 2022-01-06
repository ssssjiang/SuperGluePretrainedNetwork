from pathlib import Path
import argparse
import numpy as np

from models.utils import  AverageTimer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str, default='/persist_dataset/mower/711/711_2021-06-30-16-28_all_2021-06-30-17-10_sweep_2021-07-14-05-00-32/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_invalid_pairs', type=str, default='test/reconstruction2/invalid_matches.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_match_refine_pairs', type=str, default='test/reconstruction2/model/mis_match_image_pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--filtered_pairs', type=str, default='image_refined_invalid_matches.txt',
        help='Path to the list of filtered image pairs')
    parser.add_argument(
        '--output_dir', type=str, default='test/reconstruction2/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    opt = parser.parse_args()
    print(opt)

    with open(opt.input_root_dir + opt.input_invalid_pairs, 'r') as f:
        invalid_pairs = [l for l in f.readlines()]

    with open(opt.input_root_dir + opt.input_match_refine_pairs, 'r') as f2:
        match_refine_pairs = [l for l in f2.readlines()]

    dump_dir = Path(opt.input_root_dir + opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    output_matches_file = Path.joinpath(dump_dir, opt.filtered_pairs)

    filtered_pairs = []
    match_refine_pair_stems = []
    for i, raw_pair in enumerate(match_refine_pairs):
        pair = raw_pair.split()
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        match_refine_pair_stems.append([stem0, stem1])

    for i, raw_pair in enumerate(invalid_pairs):
        pair = raw_pair.split()
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        if [stem0, stem1] in match_refine_pair_stems or [stem1, stem0] in match_refine_pair_stems:
            filtered_pairs.append(raw_pair)

    with open(output_matches_file, 'w') as filtered:
        for pair in filtered_pairs:
            filtered.write(pair)


