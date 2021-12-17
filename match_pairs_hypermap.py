# shu.song@ninebot.com

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from copy import deepcopy

from models.matching import PureMatching
from models.utils import (quaternion_matrix, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image2,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, Loransac, MatchVerify)
# for find hloc
import sys
import os

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), "../..")))
from hloc.utils.hypermap_database import HyperMapDatabase, image_ids_to_pair_id
from hloc.utils.hfnet_database import HFNetDatabase

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_root_dir', type=str, default='/persist_dataset/mower/a4_2021-07-23-17-14_all_2021-07-23-17-57_sweep_2021-07-29-19-27-29/',
        help='Path to the root of datasets.')
    parser.add_argument(
        '--input_pairs', type=str, default='test/reconstruction/matches/matching_image_pairs.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='sensors/records_data/map/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--database', type=str, default='test/reconstruction/',
        help='Path to the hfnet.db & hypermap.db')
    parser.add_argument(
        '--output_dir', type=str, default='test/reconstruction/matches/',
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
        '--crop_size', type=int, nargs='+', default=[-1],
        help="offset_x, offset_y, width, height, if -1, do not crop")
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
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
        '--match_verify', action='store_true',
        help='match verify.')
    parser.add_argument(
        '--step_size', type=int, default=1,
        help='Set the step size of the pair to reduce the amount of '
             'test image-pairs and visualize data.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='If database exist match-pairs item, overwrite it.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.crop_size) == 4:
        print('Will crop image : offset_x = {}, offset_y = {}, width = {}, height = {}.'.format(
            opt.crop_size[0], opt.crop_size[1], opt.crop_size[2], opt.crop_size[3]))
    elif len(opt.crop_size) == 1:
        print('Will not crop images')
    else:
        raise ValueError('Cannot specify less than four integers for --crop')

    with open(opt.input_root_dir + opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda:2' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    # for debug
    matching = PureMatching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_root_dir + opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    dump_dir = Path(opt.input_root_dir + opt.output_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    output_matches_dir = Path.joinpath(dump_dir, "data")
    output_matches_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_matches_dir))
    vis_dir = Path.joinpath(dump_dir, "vis")
    vis_dir.mkdir(exist_ok=True, parents=True)
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(vis_dir))

    # Load hfnet.db and hypermap.db
    hypermap_database = str(Path(opt.input_root_dir + opt.database) / "hypermap.db")
    hfnet_database = str(Path(opt.input_root_dir + opt.database) / "hfnet.db")

    hypermap_cursor = HyperMapDatabase.connect(hypermap_database)
    hfnet_cursor = HFNetDatabase.connect(hfnet_database)

    camera_params = hypermap_cursor.read_camera_params()
    k = camera_params[0][0: 4].astype(float)
    K = np.zeros((3, 3)).astype(float)
    K[0, 0] = k[0]
    K[1, 1] = k[1]
    K[0, 2] = k[2]
    K[1, 2] = k[3]

    D = camera_params[0][4: 6].astype(float)

    # statistics average keypoints num
    all_kpts_num = []
    all_matches_num = []
    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        # Reduce test image-pairs.
        if i % opt.step_size != 0:
            continue
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_matches_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = vis_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)

        image0_id = hypermap_cursor.read_image_id_from_name(name0)
        image1_id = hypermap_cursor.read_image_id_from_name(name1)

        # Handle --cache logic.
        do_match = True
        do_viz = opt.viz
        do_match_verify = opt.match_verify
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
                # Keep the matching keypoints.
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                db_matches = np.stack([np.where(valid)[0], matches[valid]], -1)
                if len(db_matches) >= 5:
                    if opt.overwrite:
                        hypermap_cursor.replace_matches(image0_id, image1_id, db_matches)
                    else:
                        hypermap_cursor.add_matches(image0_id, image1_id, db_matches)
                else:
                    print('{}-{} merely get {} feature-matches,'
                          ' no need to add to hypermap.db.'.format(stem0, stem1, len(db_matches)))

                all_kpts_num.append((kpts0.shape[0] + kpts1.shape[0]) // 2)
                # if not do_match_verify:
                #     all_matches_num.append(len(mkpts0))
                do_match = False
            if opt.viz and viz_path.exists():
                do_viz = False
            timer.update('load_cache')

        if not (do_match or do_viz or do_match_verify):
            all_matches_num.append(len(mkpts0))
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
            kpts0 = hfnet_cursor.read_keypoints_from_image_name(name0)[:, :2]
            kpts1 = hfnet_cursor.read_keypoints_from_image_name(name1)[:, :2]

            if len(opt.crop_size) == 4:
                offset = np.zeros(2, )
                offset[0], offset[1] = opt.crop_size[0], opt.crop_size[1]
                kpts0 = kpts0 + offset[None]
                kpts1 = kpts1 + offset[None]

            scores0 = hfnet_cursor.read_keypoints_from_image_name(name0)[:, 2]
            scores1 = hfnet_cursor.read_keypoints_from_image_name(name1)[:, 2]

            descriptors0 = hfnet_cursor.read_local_descriptors_from_image_name(name0).transpose(1, 0)
            descriptors1 = hfnet_cursor.read_local_descriptors_from_image_name(name1).transpose(1, 0)

            pred = matching({'image0': inp0, 'image1': inp1,
                             'keypoints0': torch.from_numpy(kpts0.__array__()).float()[None].to(device),
                             'keypoints1': torch.from_numpy(kpts1.__array__()).float()[None].to(device),
                             'scores0': torch.from_numpy(scores0.__array__()).float()[None].to(device),
                             'scores1': torch.from_numpy(scores1.__array__()).float()[None].to(device),
                             'descriptors0': torch.from_numpy(descriptors0.__array__()).float()[None].to(device),
                             'descriptors1': torch.from_numpy(descriptors1.__array__()).float()[None].to(device)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            all_kpts_num.append((kpts0.shape[0] + kpts1.shape[0]) // 2)
            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            db_matches = np.stack([np.where(valid)[0], matches[valid]], -1)
            if (len(db_matches) >= 5):
                if opt.overwrite:
                    hypermap_cursor.replace_matches(image0_id, image1_id, db_matches)
                else:
                    hypermap_cursor.add_matches(image0_id, image1_id, db_matches)
            else:
                print('{}-{} merely get {} feature-matches,'
                      ' no need to add to hypermap.db.'.format(stem0, stem1, len(db_matches)))

        # only for test
        if len(mkpts0) and opt.match_verify:
            # LORANSAC
            th = 2.0
            n_iter = 20000
            ret, tri_angle = MatchVerify(deepcopy(mkpts0), deepcopy(mkpts1), K, K, th, n_iter, D, D)
            timer.update('ransac')
            if ret:
                db_geometries = np.stack([np.where(valid)[0][ret[2]], matches[valid][ret[2]]], -1)
                if opt.overwrite:
                    hypermap_cursor.replace_two_view_geometry(image0_id, image1_id,
                                                              db_geometries, ret[0], ret[1], tri_angle)
                else:
                    hypermap_cursor.add_two_view_geometry(image0_id, image1_id,
                                                          db_geometries, ret[0], ret[1], tri_angle)
                mask = ret[2]
            else:
                mask = np.zeros((len(mkpts0),), dtype=bool)

            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
            mconf = mconf[mask]

        all_matches_num.append(len(mkpts0))

        # Reduce visualize image data.
        if do_viz and i % (opt.step_size * 1000) == 0:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]

            # Display extra parameter info.
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    hfnet_cursor.close()
    hypermap_cursor.commit()
    hypermap_cursor.close()

    # if opt.statistic:
    print("Average number of keypoints:")
    print('Mean\t Max\t Min\t Deviation\t')
    print('{:.2f}\t {}\t {}\t {:.2f}\t'.format(np.mean(all_kpts_num), np.max(all_kpts_num), np.min(all_kpts_num),
                                               np.std(all_kpts_num)))
    print("Average number of Matches:")
    print('Mean\t Max\t Min\t Deviation\t')
    print(
        '{:.2f}\t {}\t {}\t {:.2f}\t'.format(np.mean(all_matches_num), np.max(all_matches_num), np.min(all_matches_num),
                                             np.std(all_matches_num)))
