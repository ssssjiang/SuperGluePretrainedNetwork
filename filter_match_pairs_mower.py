# shu.song@ninebot.com

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from copy import deepcopy
import seaborn as sns

from models.matching import Matching
from models.utils import (quaternion_matrix, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image2,
                          Loransac, make_distributed_plot)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str, default='/persist_dataset/mower/a4_2021-07-23-17-14_all_2021-07-23-17-57_sweep_2021-07-29-19-27-29/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_pairs', type=str, default='mower_pairs_800-360-512_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--filtered_pairs', type=str, default='filtered_lf_pairs_800-360-512.txt',
        help='Path to the list of filtered image pairs')
    parser.add_argument(
        '--output_dir', type=str, default='mower_1040_450_1024_loransac/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--eval_lf', action='store_true',
        help='Skip the pair if output .npz files are already found')


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

    # statistics average keypoints num
    all_kpts_num = []
    timer = AverageTimer(newline=True)
    filtered_pairs = []
    for i, raw_pair in enumerate(pairs):
        pair = raw_pair.split()
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem

        # lf-test
        if opt.eval_lf and stem0[-1] == stem1[-1]:
            continue

        filtered_pairs.append(raw_pair)
        matches_path = output_matches_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_evals_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)

        # Handle --cache logic.
        if opt.cache:
            # if matches_path.exists():
            #     try:
            #         results = np.load(matches_path)
            #     except:
            #         raise IOError('Cannot load matches .npz file: %s' %
            #                       matches_path)
            #
            #     kpts0, kpts1 = results['keypoints0'], results['keypoints1']
            #     matches, conf = results['matches'], results['match_confidence']
            #     all_kpts_num.append((kpts0.shape[0] + kpts1.shape[0]) // 2)

            # if opt.eval and eval_path.exists():
            #     try:
            #         results = np.load(eval_path)
            #     except:
            #         raise IOError('Cannot load eval .npz file: %s' % eval_path)
            #     err_R, err_t = results['error_R'], results['error_t']
            #     precision = results['precision']
            #     matching_score = results['matching_score']
            #     num_correct = results['num_correct']
            #     epi_errs = results['epipolar_errors']
            timer.update('load_cache')

        # if not (do_match or do_eval or do_viz or do_viz_eval):
        #     timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        #     continue

        # # Keep the matching keypoints.
        # valid = matches > -1
        # mkpts0 = kpts0[valid]
        # mkpts1 = kpts1[matches[valid]]
        # mconf = conf[valid]

    if opt.eval_lf:
        with open(opt.input_root_dir + opt.filtered_pairs, 'w') as filtered:
            for pair in filtered_pairs:
                filtered.write(pair)

    # if opt.eval:
    #     # Collate the results into a final table and print to terminal.
    #     pose_errors = []
    #     precisions = []
    #     matching_scores = []
    #     for i, pair in enumerate(pairs):
    #         if i % opt.step_size != 0:
    #             continue
    #
    #         name0, name1 = pair[:2]
    #         stem0, stem1 = Path(name0).stem, Path(name1).stem
    #
    #         if opt.eval_lf and stem0[-1] == stem0[-1]:
    #             continue
    #
    #         eval_path = output_evals_dir / \
    #                     '{}_{}_evaluation.npz'.format(stem0, stem1)
    #         results = np.load(eval_path)
    #         pose_error = np.maximum(results['error_t'], results['error_R'])
    #         pose_errors.append(pose_error)
    #         precisions.append(results['precision'])
    #         matching_scores.append(results['matching_score'])
    #     make_distributed_plot(np.array(pose_errors), dump_dir / 'pose_errors.png')
    #     thresholds = [5, 10, 20]
    #     aucs = pose_auc(pose_errors, thresholds)
    #     aucs = [100. * yy for yy in aucs]
    #     prec = 100. * np.mean(precisions)
    #     ms = 100. * np.mean(matching_scores)
    #     print('Evaluation Results (mean over {} pairs):'.format(count_pairs))
    #     print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    #     print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
    #         aucs[0], aucs[1], aucs[2], prec, ms))
    #     print("Average number of keypoints:")
    #     print('Mean\t Max\t Min\t Deviation\t')
    #     print('{:.2f}\t {}\t {}\t {:.2f}\t'.format(np.mean(all_kpts_num), np.max(all_kpts_num), np.min(all_kpts_num), np.std(all_kpts_num)))
