import os
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output_backup video file. '
             'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.53),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.25))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    it_frame = 0
    # det_file = open("./det.txt", 'w')
    statistic_file = open("./output/key_points.txt", 'w')
    while (cap.isOpened()):
        
        flag, img = cap.read()
        if not flag:
            break
        if img is None:
            continue

        img = img[int(img.shape[0] * 0.6):int(img.shape[0] * 0.85), int(img.shape[1] * 0.22):int(img.shape[1] * 0.75)]
        if args.show:
            cv2.imwrite(os.path.join(args.out_video_root,
                                     f'vis_{os.path.basename(args.video_path)}_frame_{it_frame}_input.png'), img)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        for i, pose in enumerate(pose_results):
            key_points = pose['keypoints']
            for p in key_points:
                statistic_file.write(f"{it_frame} {i} {int(p[0])} {int(p[1])}\n")
        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imwrite(os.path.join(args.out_video_root, f'vis_frame_{it_frame}.png'), vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

        it_frame += 1

    statistic_file.close()
    cap.release()
    if save_out_video:
        videoWriter.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
