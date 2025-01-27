# shu.song@ninebot.com

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2

from models.utils import (quaternion_matrix, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, make_distributed_plot)

# for find hloc
import sys
import os

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), "../..")))
from hloc.utils.hypermap_database import HyperMapDatabase, image_ids_to_pair_id
# from hloc.utils.hfnet_database import HFNetDatabase

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with hfnet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str, default='/persist_dataset/mower/B6_2021-06-30-10-45_all_2021-06-30-12-20_sweep_2021-07-14-06-10-39/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_pairs', type=str, default='test/reconstruction2/model/mis_match_feature_pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='sensors/records_data/map',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--database', type=str, default='test/reconstruction2/',
        help='Path to the hfnet.db & hypermap.db')
    parser.add_argument(
        '--output_dir', type=str, default='test/reconstruction2/filtered/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='jpg', choices=['jpg', 'png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--step_size', type=int, default=1,
        help='Set the step size of the pair to reduce the amount of '
             'test image-pairs and visualize data.')
    # parser.add_argument(
    #     '--eval_R_err', action='store_true',
    #     help='.')
    # parser.add_argument(
    #     '--eval_t_err', action='store_true',
    #     help='.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    with open(opt.input_root_dir + opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 21 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 21 valid entries per row'.format(opt.input_pairs))

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_root_dir + opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    dump_dir = Path(opt.input_root_dir + opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
   # output_matches_dir = Path.joinpath(dump_dir, "data", "matches")
    # cache match_pairs_hypermap
    output_matches_dir = Path.joinpath(dump_dir, "data")
    output_matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_matches_dir))
    output_evals_dir = Path.joinpath(dump_dir, "data", "evals")
    output_evals_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = Path.joinpath(dump_dir, "vis_mis_feature_pairs")
    vis_dir.mkdir(exist_ok=True, parents=True)
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_evals_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(vis_dir))

    # Load hfnet.db and hypermap.db
    hypermap_database0 = str(Path(opt.input_root_dir + opt.database) / "bak/hypermap-E.db")
    # hfnet_database = str(Path(opt.input_root_dir + opt.database) / "hfnet.db")
    hypermap_database1 = str(Path(opt.input_root_dir + opt.database) / "hypermap.db")

    hypermap_cursor0 = HyperMapDatabase.connect(hypermap_database0)
    hypermap_cursor1 = HyperMapDatabase.connect(hypermap_database1)
    # hfnet_cursor = HFNetDatabase.connect(hfnet_database)

    # statistics average keypoints num
    all_kpts_num = []
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        # Reduce test image-pairs.
        if i % opt.step_size != 0:
            continue
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_matches_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        nn_viz_path = vis_dir / '{}_{}_nn_filter_matches.{}'.format(stem0, stem1, opt.viz_extension)
        sfm_viz_path = vis_dir / '{}_{}_sfm_filter_matches.{}'.format(stem0, stem1, opt.viz_extension)
        # Handle --cache logic.
        do_match = True
        do_viz = opt.viz

        # Load the image pair.
        image0 = cv2.imread(str(input_dir / name0), cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(str(input_dir / name1), cv2.IMREAD_GRAYSCALE)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir / name0, input_dir / name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            image0_id = hypermap_cursor0.read_image_id_from_name(name0)
            image1_id = hypermap_cursor0.read_image_id_from_name(name1)

            pair_id = image_ids_to_pair_id(image0_id, image1_id)
            raw_matches = hypermap_cursor0.read_matches_from_pair_id(pair_id)
            filter_matches = hypermap_cursor1.read_matches_from_pair_id(pair_id)

            kpts0 = hypermap_cursor0.read_keypoints_from_image_id(image0_id)[:, 0:2]
            kpts1 = hypermap_cursor0.read_keypoints_from_image_id(image1_id)[:, 0:2]

            # matches = np.full((max(np.shape(kpts0)[0], np.shape(kpts1)[0]),), -1)
            matches0 = np.full((np.shape(kpts0)[0],), -1)
            matches1 = np.full((np.shape(kpts0)[0],), -1)
            if raw_matches is not None:
                for match in raw_matches:
                    matches0[match[0]] = match[1]
            if filter_matches is not None:
                for match in filter_matches:
                    matches1[match[0]] = match[1]
            timer.update('matcher')

            all_kpts_num.append((kpts0.shape[0] + kpts1.shape[0]) // 2)
            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches_nn': matches0, 'matches_filter':matches1}
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        # valid_nn = matches0 > -1
        # valid_filter = matches1 > -1
        nn_filter = (matches0 == -1) & (matches1 > -1)
        sfm_filter = (matches0 > -1) & (matches1 == -1)
        mkpts0_nn = kpts0[nn_filter]
        mkpts1_nn = kpts1[matches1[nn_filter]]
        mconf_nn = np.full((np.shape(mkpts0_nn)[0],), 0.1)
        mkpts0_sfm = kpts0[sfm_filter]
        mkpts1_sfm = kpts1[matches0[sfm_filter]]
        mconf_sfm = np.full((np.shape(mkpts0_sfm)[0],), 0.1)

        # Reduce visualize image data.
        if do_viz and i % (opt.step_size * 1000) == 0:
            # Visualize the nn filtered matches.
            color_nn = cm.jet(mconf_nn)
            text = [
                'global filtered & nn pass',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0_nn)),
            ]

            # Display extra parameter info.
            small_text = [
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0_nn, mkpts1_nn, color_nn,
                text, nn_viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            # Visualize the global sfm filtered matches.
            color_sfm = cm.jet(mconf_sfm)
            text = [
                'nn filtered & global pass',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0_sfm)),
            ]

            # Display extra parameter info.
            small_text = [
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0_sfm, mkpts1_sfm, color_sfm,
                text, sfm_viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)
            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

