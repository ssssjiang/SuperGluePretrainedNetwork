# shu.song@ninebot.com

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from copy import deepcopy

from models.matching import Matching
from models.utils import (quaternion_matrix, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image2,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, Loransac)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/mower_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/map/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='/persist_dataset/SuperGlue_result/mower_800_360/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1280, 720],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--crop_size', type=int, nargs='+', default=[240, 0, 800, 360],
        help="offset_x, offset_y, width, height, if -1, do not crop")
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=50,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

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
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--loransac', action='store_true',
        help='loransac.')
    parser.add_argument(
        '--step_size', type=int, default=1,
        help='Set the step size of the pair to reduce the amount of '
             'test image-pairs and visualize data.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    # if len(opt.resize) == 2 and opt.resize[1] == -1:
    #     opt.resize = opt.resize[0:1]
    # if len(opt.resize) == 2:
    #     print('Will resize to {}x{} (WxH)'.format(
    #         opt.resize[0], opt.resize[1]))
    # elif len(opt.resize) == 1 and opt.resize[0] > 0:
    #     print('Will resize max dimension to {}'.format(opt.resize[0]))
    # elif len(opt.resize) == 1:
    #     print('Will not resize images')
    # else:
    #     raise ValueError('Cannot specify more than two integers for --resize')

    if len(opt.crop_size) == 4:
        print('Will crop image : offset_x = {}, offset_y = {}, width = {}, height = {}.'.format(
            opt.crop_size[0], opt.crop_size[1], opt.crop_size[2], opt.crop_size[3]))
    elif len(opt.crop_size) == 1:
        print('Will not crop images')
    else:
        raise ValueError('Cannot specify less than four integers for --crop')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 21 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda:2' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    # for debug
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    dump_dir = Path(opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    output_matches_dir = Path.joinpath(dump_dir, "data", "matches")
    output_matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_matches_dir))
    output_evals_dir = Path.joinpath(dump_dir, "data", "evals")
    output_evals_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = Path.joinpath(dump_dir, "vis")
    vis_dir.mkdir(exist_ok=True, parents=True)
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_evals_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(vis_dir))

    # statistics average keypoints num
    all_kpts_num = 0
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
        # miao
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                all_kpts_num = ((kpts0.shape(0) + kpts1.shape(0)) // 2) + all_kpts_num
                do_match = False
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
        image0, inp0 = read_image2(
            input_dir / name0, device, 0, opt.crop_size)
        image1, inp1 = read_image2(
            input_dir / name1, device, 0, opt.crop_size)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir / name0, input_dir / name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            if len(opt.crop_size) == 4:
                offset = np.zeros(2, )
                offset[0], offset[1] = opt.crop_size[0], opt.crop_size[1]
                kpts0 = kpts0 + offset[None]
                kpts1 = kpts1 + offset[None]
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            all_kpts_num = ((kpts0.shape(0) + kpts1.shape(0)) // 2) + all_kpts_num

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 21, 'Pair does not have ground truth info'
            k0 = np.array(pair[2: 6]).astype(float)
            K0 = np.zeros((3, 3)).astype(float)
            K1 = np.zeros((3, 3)).astype(float)
            K0[0, 0] = k0[0]
            K0[1, 1] = k0[1]
            K0[0, 2] = k0[2]
            K0[1, 2] = k0[3]
            k1 = np.array(pair[8: 12]).astype(float)
            K1[0, 0] = k1[0]
            K1[1, 1] = k1[1]
            K1[0, 2] = k1[2]
            K1[1, 2] = k1[3]

            D0 = np.array(pair[6: 8]).astype(float)
            D1 = np.array(pair[12: 14]).astype(float)
            q_0to1 = np.array(pair[14: 18]).astype(float)
            t_0to1 = np.array(pair[18:]).astype(float)
            T_0to1 = quaternion_matrix(q_0to1)
            T_0to1[0: 3, 3] = t_0to1

            # # Scale the intrinsics to resized image.
            # K0 = scale_intrinsics(K0, scales0)
            # K1 = scale_intrinsics(K1, scales1)

            if opt.loransac:
                # LORANSAC
                th = 2.0
                n_iter = 20000
                mask = Loransac(deepcopy(mkpts0), deepcopy(mkpts1), K0, K1, th, n_iter, D0, D1)
            else:
                mask = np.ones((len(mkpts0),), dtype=bool)
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
            mconf = mconf[mask]
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1, D0, D1)

            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / min(len(kpts0), len(kpts1)) if min(len(kpts0), len(kpts1)) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh, D0=D0, D1=D1)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
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
        if do_viz and i % (opt.step_size * 16) == 0:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        if do_viz_eval and i % (opt.step_size * 16) == 0:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]

            # Display extra parameter info (only works with --fast_viz).
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
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
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        prec = 100. * np.mean(precisions)
        ms = 100. * np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))
        print("Average number of keypoints : {}\n".format(all_kpts_num // len(pairs)))
