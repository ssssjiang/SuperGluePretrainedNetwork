# data from hypermap.db - two view geometries
# hypermap data + eval(error_R & error_t)

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import cv2

from models.utils import (quaternion_matrix, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, make_distributed_plot)

# for find hloc
import sys
import os

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), "../..")))
from hloc.utils.hypermap_database import HyperMapDatabase, image_ids_to_pair_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with hfnet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str,
        default='/persist_dataset/mower/raptor_record/A4_2021-12-15_yk_kapture_mapping/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_pairs', type=str, default='reconstruction/valid_matches_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='sensors/records_data/map',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--database', type=str, default='/reconstruction/',
        help='Path to the hfnet.db & hypermap.db')
    parser.add_argument(
        '--output_dir', type=str, default='reconstruction/global_model/croped/match/',
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
        if not all([len(p) == 17 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 17 valid entries per row'.format(opt.input_pairs))

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_root_dir + opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    dump_dir = Path(opt.input_root_dir + opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    output_matches_dir = Path.joinpath(dump_dir, "data", "matches")
    output_matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_matches_dir))
    output_evals_dir = Path.joinpath(dump_dir, "data", "evals")
    output_evals_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = Path.joinpath(dump_dir, "vis_after_match_refine")
    vis_dir.mkdir(exist_ok=True, parents=True)
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_evals_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(vis_dir))

    # Load hfnet.db and hypermap.db
    hypermap_database = str(Path(opt.input_root_dir + opt.database) / "hypermap.db")

    hypermap_cursor = HyperMapDatabase.connect(hypermap_database)

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
        eval_path = output_evals_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = vis_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = vis_dir / \
                        '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

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
            image0_id = hypermap_cursor.read_image_id_from_name(name0)
            image1_id = hypermap_cursor.read_image_id_from_name(name1)

            pair_id = image_ids_to_pair_id(image0_id, image1_id)
            print(pair_id)
            raw_matches = hypermap_cursor.read_matches_from_pair_id(pair_id)

            if raw_matches is None:
                continue

            q = hypermap_cursor.read_q_from_pair_id(pair_id)
            t = hypermap_cursor.read_t_from_pair_id(pair_id)
            config = hypermap_cursor.read_config_from_pair_id(pair_id)
            if q is not None and t is not None:
                R = quaternion_matrix(q)[:3, :3]
                t = t[:, 0]
            else:
                R = None
                t = None

            kpts0 = hypermap_cursor.read_keypoints_from_image_id(image0_id)[:, 0:2]
            kpts1 = hypermap_cursor.read_keypoints_from_image_id(image1_id)[:, 0:2]

            # matches = np.full((max(np.shape(kpts0)[0], np.shape(kpts1)[0]),), -1)
            matches = np.full((np.shape(kpts0)[0],), -1)
            if raw_matches is not None:
                for match in raw_matches:
                    matches[match[0]] = match[1]
            timer.update('matcher')

            all_kpts_num.append((kpts0.shape[0] + kpts1.shape[0]) // 2)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = np.full((np.shape(mkpts0)[0],), 0.5)

        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 17, 'Pair does not have ground truth info'
            k0 = np.array(pair[2: 6]).astype(float)
            K0 = np.zeros((3, 3)).astype(float)
            K1 = np.zeros((3, 3)).astype(float)
            K0[0, 0] = k0[0]
            K0[1, 1] = k0[1]
            K0[0, 2] = k0[2]
            K0[1, 2] = k0[3]
            k1 = np.array(pair[6: 10]).astype(float)
            K1[0, 0] = k1[0]
            K1[1, 1] = k1[1]
            K1[0, 2] = k1[2]
            K1[1, 2] = k1[3]

            # D0 = np.array(pair[6: 8]).astype(float)
            # D1 = np.array(pair[12: 14]).astype(float)
            q_0to1 = np.array(pair[10: 14]).astype(float)
            t_0to1 = np.array(pair[14:]).astype(float)
            T_0to1 = quaternion_matrix(q_0to1)
            T_0to1[0: 3, 3] = t_0to1

            # # Scale the intrinsics to resized image.
            # K0 = scale_intrinsics(K0, scales0)
            # K1 = scale_intrinsics(K1, scales1)

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)

            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / min(len(kpts0), len(kpts1)) if min(len(kpts0), len(kpts1)) > 0 else 0

            # thresh = 1.  # In pixels relative to resized image size.
            # ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh, D0=D0, D1=D1)
            # if ret is None:
            #     err_t, err_R = np.inf, np.inf
            # else:
            #     R, t, inliers = ret
            #     err_t, err_R = compute_pose_error(T_0to1, R, t)
            if R is None:
                err_t, err_R = np.inf, np.inf
            else:
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        # Reduce visualize image data.
        # if do_viz and i % (opt.step_size * 100) == 0:
        #     # Visualize the matches.
        #     color = cm.jet(mconf)
        #     text = [
        #         'Hfnet',
        #         'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        #         'Matches: {}'.format(len(mkpts0)),
        #     ]
        #
        #     # Display extra parameter info.
        #     small_text = [
        #         'Image Pair: {}:{}'.format(stem0, stem1),
        #     ]
        #
        #     make_matching_plot(
        #         image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        #         text, viz_path, opt.show_keypoints,
        #         opt.fast_viz, opt.opencv_display, 'Matches', small_text)
        #
        #     timer.update('viz_match')

        inlier_ratio = num_correct / (matches > -1).sum()
        show_viz = len(matches) > 30 and \
                   (i % (opt.step_size * 1000) == 0 or err_t > 10 or err_R > 3 or inlier_ratio < 0.8)
        if do_viz_eval and show_viz:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'Hfnet Config: {}'.format(config),
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]

            # Display extra parameter info (only works with --fast_viz).
            small_text = [
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)
            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        R_errors = []
        t_errors = []
        precisions = []
        matching_scores = []
        for i, pair in enumerate(pairs):
            if i % opt.step_size != 0:
                continue

            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_evals_dir / \
                        '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            R_error = results['error_R']
            t_error = results['error_t']
            pose_errors.append(pose_error)
            R_errors.append(R_error)
            t_errors.append(t_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        # save pose errors
        output_pose_err = dump_dir / 'pose_errors.npz'
        np.savez(str(output_pose_err), pose_errors=pose_errors, R_errors=R_errors, t_errors=t_errors)
        # make_distributed_plot(np.array(pose_errors), dump_dir / 'pose_errors.png')
        # make_distributed_plot(np.array(R_errors), dump_dir / 'R_errors.png')
        # make_distributed_plot(np.array(t_errors), dump_dir / 't_errors.png')
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        R_aucs = pose_auc(R_errors, thresholds)
        t_aucs = pose_auc(t_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        R_aucs = [100. * yy for yy in R_aucs]
        t_aucs = [100. * yy for yy in t_aucs]
        prec = 100. * np.mean(precisions)
        ms = 100. * np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))
        print('R_AUC@5\t R_AUC@10\t R_AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(
            R_aucs[0], R_aucs[1], R_aucs[2]))
        print('t_AUC@5\t t_AUC@10\t t_AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(
            t_aucs[0], t_aucs[1], t_aucs[2]))
        print("Average number of keypoints:")
        print('Mean\t Max\t Min\t Deviation\t')
        print('{:.2f}\t {}\t {}\t {:.2f}\t'.format(np.mean(all_kpts_num), np.max(all_kpts_num), np.min(all_kpts_num),
                                                   np.std(all_kpts_num)))
