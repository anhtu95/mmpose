import numpy as np
from mmcv import Config


class DatasetInfo:

    def __init__(self, dataset_info_file):
        self.cfg = Config.fromfile(dataset_info_file)

        self.dataset_info = self.cfg._cfg_dict['dataset_info']
        self.dataset_name = self.dataset_info['dataset_name']
        self.paper_info = self.dataset_info['paper_info']
        self.keypoint_info = self.dataset_info['keypoint_info']
        self.skeleton_info = self.dataset_info['skeleton_info']
        self.joint_weights = self.dataset_info['joint_weights']
        self.sigmas = self.dataset_info['sigmas']

        self._parse_keypoint_info()
        self._parse_skeleton_info()

    def _parse_skeleton_info(self):
        """Parse skeleton info.

        - limb_num (int): number of limbs.
        - skeleton (list((2,))): list of limbs (id).
        - skeleton_name (list((2,))): list of limbs (name).
        - pose_limb_color (np.ndarray): the color of the limb for
            visualization.
        """
        self.limb_num = len(self.skeleton_info.keys())
        self.pose_limb_color = []

        self.skeleton_name = []
        self.skeleton = []
        for skid in self.skeleton_info.keys():
            limb = self.skeleton_info[skid]['limb']
            self.skeleton_name.append(limb)
            self.skeleton.append([
                self.keypoint_name2id[limb[0]], self.keypoint_name2id[limb[1]]
            ])
            self.pose_limb_color.append(self.skeleton_info[skid].get(
                'color', [255, 128, 0]))
        self.pose_limb_color = np.array(self.pose_limb_color)

    def _parse_keypoint_info(self):
        """Parse keypoint info.

        - keypoint_num (int): number of keypoints.
        - keypoint_id2name (dict): mapping keypoint id to keypoint name.
        - keypoint_name2id (dict): mapping keypoint name to keypoint id.
        - upper_body_ids (list): a list of keypoints that belong to the
            upper body.
        - lower_body_ids (list): a list of keypoints that belong to the
            lower body.
        - flip_index (list): list of flip index (id)
        - flip_pairs (list((2,))): list of flip pairs (id)
        - flip_index_name (list): list of flip index (name)
        - flip_pairs_name (list((2,))): list of flip pairs (name)
        - pose_kpt_color (np.ndarray): the color of the keypoint for
            visualization.
        """

        self.keypoint_num = len(self.keypoint_info.keys())
        self.keypoint_id2name = {}
        self.keypoint_name2id = {}

        self.pose_kpt_color = []
        self.upper_body_ids = []
        self.lower_body_ids = []

        self.flip_index_name = []
        self.flip_pairs_name = []

        for kid in self.keypoint_info.keys():

            keypoint_name = self.keypoint_info[kid]['name']
            self.keypoint_id2name[kid] = keypoint_name
            self.keypoint_name2id[keypoint_name] = kid
            self.pose_kpt_color.append(self.keypoint_info[kid].get(
                'color', [255, 128, 0]))

            type = self.keypoint_info[kid].get('type', '')
            if type == 'upper':
                self.upper_body_ids.append(kid)
            elif type == 'lower':
                self.lower_body_ids.append(kid)
            else:
                pass

            swap_keypoint = self.keypoint_info[kid].get('swap', '')
            if swap_keypoint == keypoint_name or swap_keypoint == '':
                self.flip_index_name.append(keypoint_name)
            else:
                self.flip_index_name.append(swap_keypoint)
                if (swap_keypoint, keypoint_name) not in self.flip_pairs_name:
                    self.flip_pairs_name.append([keypoint_name, swap_keypoint])

        self.flip_pairs = [[
            self.keypoint_name2id[pair[0]], self.keypoint_name2id[pair[1]]
        ] for pair in self.flip_pairs_name]
        self.flip_index = [
            self.keypoint_name2id[name] for name in self.flip_index_name
        ]
        self.pose_kpt_color = np.array(self.pose_kpt_color)