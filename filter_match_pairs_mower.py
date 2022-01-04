# shu.song@ninebot.com

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from copy import deepcopy
import seaborn as sns

from models.utils import  AverageTimer

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str, default='/persist_dataset/mower/B6_2021-06-30-10-45_all_2021-06-30-12-20_sweep_2021-07-14-06-10-39/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_pairs', type=str, default='test/reconstruction3/mower_pairs_800-360-1024_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--filtered_pairs', type=str, default='test/reconstruction3/lf_pairs_gt.txt',
        help='Path to the list of filtered image pairs')
    parser.add_argument(
        '--output_dir', type=str, default='test/reconstruction3/matches/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--output_dir2', type=str, default='test/reconstruction2/mower_sp_800_360_1024/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--eval_lf', action='store_true',
        help='Filter front-left match pairs.')
    parser.add_argument(
        '--eval_err', action='store_true',
        help='filter high pose error.')
    parser.add_argument(
        '--eval_err_compare', action='store_true',
        help='filter high pose error.')


    opt = parser.parse_args()
    print(opt)

    with open(opt.input_root_dir + opt.input_pairs, 'r') as f:
        pairs = [l for l in f.readlines()]


    # if opt.eval:
    #     if not all([len(p) == 21 for p in pairs]):
    #         raise ValueError(
    #             'All pairs should have ground truth info for evaluation.'
    #             'File \"{}\" needs 21 valid entries per row'.format(opt.input_pairs))

    # Create the output directories if they do not exist already.
    dump_dir = Path(opt.input_root_dir + opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    output_matches_dir = Path.joinpath(dump_dir, "data", "matches")
    output_matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_matches_dir))
    output_evals_dir = Path.joinpath(dump_dir, "data", "evals")
    output_evals_dir.mkdir(exist_ok=True, parents=True)

    # for Compared
    if opt.eval_err_compare:
        output_evals_dir2 = Path.joinpath(Path(opt.input_root_dir + opt.output_dir2), "data", "evals")

    # statistics average keypoints num
    all_kpts_num = []
    timer = AverageTimer(newline=True)
    filtered_pairs = []
    for i, raw_pair in enumerate(pairs):
        pair = raw_pair.split()
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem

        matches_path = output_matches_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_evals_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)

        # lf-test
        if opt.eval_lf:
            if stem0[-1] == stem1[-1]:
                continue
            else:
                filtered_pairs.append(raw_pair)
                timer.update('load_cache')

        if opt.eval_err_compare:
            eval_path2 = output_evals_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        # Handle --cache logic.

        if opt.eval_err and eval_path.exists():
            try:
                results = np.load(eval_path)
            except:
                raise IOError('Cannot load eval .npz file: %s' % eval_path)
            err_R, err_t = results['error_R'], results['error_t']
            precision = results['precision']
            matching_score = results['matching_score']
            num_correct = results['num_correct']
            epi_errs = results['epipolar_errors']

            if opt.eval_err_compare:
                try:
                    results2 = np.load(eval_path2)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path2)
                err_R2, err_t2 = results2['error_R'], results2['error_t']
                if err_t > 20 or err_t2 > 20:
                    filtered_pairs.append(raw_pair)
            elif err_t > 20:
                filtered_pairs.append(raw_pair)
            timer.update('load_cache')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

        # if not (do_match or do_eval or do_viz or do_viz_eval):
        #     timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        #     continue

        # # Keep the matching keypoints.
        # valid = matches > -1
        # mkpts0 = kpts0[valid]
        # mkpts1 = kpts1[matches[valid]]
        # mconf = conf[valid]

    with open(opt.input_root_dir + opt.filtered_pairs, 'w') as filtered:
        for pair in filtered_pairs:
            filtered.write(pair)
