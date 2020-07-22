from config.config import config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from layer.model import EfficientNet
import torch.nn.functional as F


class TrackUtil:
    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        boxes = np.concatenate([detection[:, :2] + detection[:, 2:] / 2, detection[:, 2:] * config['crop_delta']], axis=1)
        return boxes

    @staticmethod
    def convert_image(image, detection):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        h, w, _ = image.shape
        size = config['crop_size']
        image = image.astype(np.float32)
        image -= TrackerConfig.mean_pixel
        objects = np.zeros((1, size, size, 3))
        for box in detection:
            # box = boxes[i]
            width = int(box[2])
            height = int(box[3])
            if width == 0 or height == 0:
                ob = np.zeros((1, size, size, 3))
            else:
                center_x = int(box[0])
                center_y = int(box[1])
                y0 = center_y - height // 2
                y1 = center_y + height // 2
                x0 = center_x - width // 2
                x1 = center_x + width // 2
                # first version 关于边缘目标的处理错误
                if y0 < 0:
                    y0 = 0
                elif y1 > h:
                    y1 = h
                if x0 < 0:
                    x0 = 0
                elif x1 > w:
                    x1 = w
                ob = image[y0: y1, x0: x1]
                ob = cv2.resize(ob, (size, size))
                ob = np.expand_dims(ob, axis=0)
            objects = np.concatenate([objects, ob], 0)
        objects = torch.from_numpy(objects[1:]).permute(0, 3, 1, 2)
        if TrackerConfig.cuda:
            return Variable(objects.cuda())
        return Variable(objects)

    @staticmethod
    def get_iou(pre_boxes, next_boxes):
        h = len(pre_boxes)
        w = len(next_boxes)
        if h == 0 or w == 0:
            return []

        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            b1 = np.copy(pre_boxes[i, :])
            # b1[2:] = b1[:2] + b1[2:]
            for j in range(w):
                b2 = np.copy(next_boxes[j, :])
                # b2[2:] = b2[:2] + b2[2:]
                delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
                delta_w = min(b1[3], b2[3]) - max(b1[1], b2[1])
                if delta_h < 0 or delta_w < 0:
                    expand_area = (max(b1[2], b2[2]) - min(b1[0], b2[0])) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))
                    area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou[i, j] = -(expand_area - area) / area
                else:
                    overlap = delta_h * delta_w
                    area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - max(overlap, 0)
                    iou[i, j] = overlap / area

        return iou

    @staticmethod
    def get_node_similarity(n1, n2, frame_index, recorder):
        if n1.frame_index > n2.frame_index:
            n_max = n1
            n_min = n2
        elif n1.frame_index < n2.frame_index:
            n_max = n2
            n_min = n1
        else:  # in the same frame_index
            return None

        f_max = n_max.frame_index
        f_min = n_min.frame_index

        # not recorded in recorder
        if frame_index - f_min >= TrackerConfig.max_track_node:
            return None

        return recorder.all_similarity[f_max][f_min][n_min.id, n_max.id]

    @staticmethod
    def get_merge_similarity(t1, t2, frame_index, recorder):
        '''
        Get the similarity between two tracks
        :param t1: track 1
        :param t2: track 2
        :param frame_index: current frame_index
        :param recorder: recorder
        :return: the similairty (float value). if valid, return None
        '''
        merge_value = []
        if t1 is t2:
            return None

        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f1 in enumerate(all_f1):
            for j, f2 in enumerate(all_f2):
                compare_f = [f1 + 1, f1 - 1]
                for f in compare_f:
                    if f not in all_f1 and f == f2:
                        n1 = t1.nodes[i]
                        n2 = t2.nodes[j]
                        s = TrackUtil.get_node_similarity(n1, n2, frame_index, recorder)
                        if s is None:
                            continue
                        merge_value += [s]

        if len(merge_value) == 0:
            return None
        return np.mean(np.array(merge_value))

    @staticmethod
    def merge(t1, t2):
        '''
        merge t2 to t1, after that t2 is set invalid
        :param t1: track 1
        :param t2: track 2
        :return: None
        '''
        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f2 in enumerate(all_f2):
            if f2 not in all_f1:
                insert_pos = 0
                for j, f1 in enumerate(all_f1):
                    if f2 < f1:
                        break
                    insert_pos += 1
                t1.nodes.insert(insert_pos, t2.nodes[i])

        # remove some nodes in t1 in order to keep t1 satisfy the max nodes
        if len(t1.nodes) > TrackerConfig.max_track_node:
            t1.nodes = t1.nodes[-TrackerConfig.max_track_node:]
        t1.age = min(t1.age, t2.age)
        t2.valid = False

    @staticmethod
    def ECC(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, eps=1e-5,
            max_iter=100, scale=[200, 200], align=False):
        """Compute the warp matrix from src to dst.
        Parameters
        ----------
        src : ndarray
            An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
        dst : ndarray
            An NxM matrix of target img(BGR or Gray).
        warp_mode: flags of opencv
            translation: cv2.MOTION_TRANSLATION
            rotated and shifted: cv2.MOTION_EUCLIDEAN
            affine(shift,rotated,shear): cv2.MOTION_AFFINE
            homography(3d): cv2.MOTION_HOMOGRAPHY
        eps: float
            the threshold of the increment in the correlation coefficient between two iterations
        max_iter: int
            the number of iterations.
        scale: float or [int, int]
            scale_ratio: float
            scale_size: [W, H]
        align: bool
            whether to warp affine or perspective transforms to the source image
        Returns
        -------
        warp matrix : ndarray
            Returns the warp matrix from src to dst.
            if motion models is homography, the warp matrix will be 3x3, otherwise 2x3
        src_aligned: ndarray
            aligned source image of gray
        """
        assert src.shape == dst.shape, "the source image must be the same format to the target image!"

        # BGR2GRAY
        if src.ndim == 3:
            # Convert images to grayscale
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        # make the imgs smaller to speed up
        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                if scale != 1:
                    src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    scale = [scale, scale]
                else:
                    src_r, dst_r = src, dst
                    scale = None
            else:
                if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                    src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
                else:
                    src_r, dst_r = src, dst
                    scale = None
        else:
            src_r, dst_r = src, dst

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
        except:
            return None, None

        if scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        if align:
            sz = src.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
            return warp_matrix, src_aligned
        else:
            return warp_matrix, None

    @staticmethod
    def AffinePoints(points, warp_matrix, scale=None):
        """Compute the warp matrix from src to dst.
        Parameters
        ----------
        points : array like
            An Nx2 matrix of N points
        warp_matrix : ndarray
            An 2x3 or 3x3 matrix of warp_matrix.
        scale: float or [int, int]
            scale_ratio: float
            scale_x,scale_y: [float, float]
            scale = (image size of ECC) / (image size now)
            if the scale is not None, which means the transition factor in warp matrix must be multiplied
        Returns
        -------
        warped points : ndarray
            Returns an Nx2 matrix of N aligned points
        """
        points = points.reshape(-1, 2)
        points = np.asarray(points)
        if points.ndim == 1:
            points = points[np.newaxis, :]
        assert points.shape[1] == 2, 'points need (x,y) coordinate'

        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                scale = [scale, scale]
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        v = np.ones((points.shape[0], 1))
        points = np.c_[points, v]  # [x, y] -> [x, y, 1], Nx3

        aligned_points = warp_matrix @ points.T
        aligned_points = aligned_points[:2, :].T

        return aligned_points.reshape(-1, 4)


class TrackerConfig:
    """
    max_record_frame = 30
    max_track_age = 30
    max_track_node = 30
    max_draw_track_node = 30
    """
    max_record_frame = 25
    max_track_age = 25
    max_track_node = 25
    max_draw_track_node = 25

    max_object = config['max_object']
    sst_model_path = config['resume']
    cuda = config['cuda']
    mean_pixel = config['mean_pixel']
    image_size = (config['sst_dim'], config['sst_dim'])

    min_iou_frame_gap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_iou = [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0]
    # min_iou = [pow(0.3, i) for i in min_iou_frame_gap]

    min_merge_threshold = 0.9

    max_bad_node = 0.9

    decay = 0.995

    roi_verify_max_iteration = 2
    roi_verify_punish_rate = 0.6

    @staticmethod
    def set_configure(all_choice):
        min_iou_frame_gaps = [
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        min_ious = [
            # [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
            [0.3, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.2, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [-1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
        ]

        decays = [1 - 0.01 * i for i in range(11)]

        roi_verify_max_iterations = [2, 3, 4, 5, 6]

        roi_verify_punish_rates = [0.6, 0.4, 0.2, 0.1, 0.0, 1.0]

        max_track_ages = [i * 3 for i in range(1, 11)]
        max_track_nodes = [i * 3 for i in range(1, 11)]

        if all_choice is None:
            return
        # all_choice = (0, 0, 4, 0, 3, 3)
        TrackerConfig.min_iou_frame_gap = min_iou_frame_gaps[all_choice[0]]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        TrackerConfig.min_iou = min_ious[all_choice[0]]
        # [0.3, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
        TrackerConfig.decay = decays[all_choice[1]]
        # 1
        TrackerConfig.roi_verify_max_iteration = roi_verify_max_iterations[all_choice[2]]
        # 6
        TrackerConfig.roi_verify_punish_rate = roi_verify_punish_rates[all_choice[3]]
        # 0.6
        TrackerConfig.max_track_age = max_track_ages[all_choice[4]]
        # 12
        TrackerConfig.max_track_node = max_track_nodes[all_choice[5]]
        # 12

    @staticmethod
    def get_configure_str(all_choice):
        return "{}_{}_{}_{}_{}_{}".format(all_choice[0], all_choice[1], all_choice[2], all_choice[3], all_choice[4],
                                          all_choice[5])

    @staticmethod
    def get_all_choices():
        # return [(1, 1, 0, 0, 4, 2)]
        return [(i1, i2, i3, i4, i5, i6) for i1 in range(5) for i2 in range(5) for i3 in range(5) for i4 in range(5) for
                i5 in range(5) for i6 in range(5)]

    @staticmethod
    def get_all_choices_decay():
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_all_choices_max_track_node():
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_choices_age_node():
        return [(0, 0, 0, 0, a, n) for a in range(10) for n in range(10)]

    @staticmethod
    def get_ua_choice():
        return (5, 0, 4, 1, 5, 5)


class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self):
        self.max_record_frame = TrackerConfig.max_record_frame
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}
        self.all_iou = {}
        self.all_image = {}

    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size
        new_data = Variable(torch.ones(shape) * constant).type(torch.DoubleTensor)
        if config['cuda']:
            new_data = new_data.cuda()
        return torch.cat([x, new_data], dim=dim)

    def euclidean_dist(self, x, y):
        square_x = x.pow(2)
        square_x = square_x.sum(-1)
        square_y = y.pow(2)
        square_y = square_y.sum(-1)
        ex = square_x.unsqueeze(-1)
        ey = square_y.unsqueeze(-2)
        xy = torch.einsum('bij,bkj->bik', x, y)
        exy = ex - 2 * xy + ey + 1e-10
        dist = exy.sqrt()
        return dist

    def get_similar(self, pre_feature, cur_feature, fill_up_column=True):
        pre_num = pre_feature.shape[1]
        next_num = cur_feature.shape[1]
        dist = self.euclidean_dist(pre_feature, cur_feature)
        x = dist.squeeze()
        x = self.resize_dim(x, 1, dim=0, constant=config['false_constant'])
        x = self.resize_dim(x, 1, dim=1, constant=config['false_constant'])
        x = 1 / (x + 0.01)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        x = Variable(torch.zeros(pre_num, next_num + 1))
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        x[:, next_num:next_num + 1] = x_f[:pre_num, next_num:next_num + 1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, next_num:next_num + 1].repeat(1, pre_num - 1)], dim=1)

        if config['cuda']:
            y = x.data.cpu().numpy()
            # del x, x_f, x_t
            # torch.cuda.empty_cache()
        else:
            y = x.data.numpy()
        return y

    def update(self, frame_index, features, boxes, image):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                del self.all_image[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            self.all_image[frame_index] = image

            self.all_similarity[frame_index] = {}
            if frame_index == 4:
                exit(0)
            for pre_index in self.all_frame_index[:-1]:
                delta = pow(TrackerConfig.decay, (frame_index - pre_index) / 3.0)
                pre_similarity = self.get_similar(Variable(self.all_features[pre_index]),
                                                  Variable(features), fill_up_column=False)
                print(pre_similarity)
                self.all_similarity[frame_index][pre_index] = pre_similarity * delta

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                warp_matrix, _ = TrackUtil.ECC(self.all_image[pre_index], image)
                if warp_matrix is None:
                    iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                else:
                    aligned_boxes = TrackUtil.AffinePoints(self.all_boxes[pre_index], warp_matrix)
                    iou = TrackUtil.get_iou(aligned_boxes, boxes)
                print(iou)
                self.all_iou[frame_index][pre_index] = iou
        else:
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            self.all_image[frame_index] = image
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index + 1]:
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = self.euclidean_dist(Variable(self.all_features[pre_index]),
                                                     Variable(self.all_features[-1]))
                self.all_similarity[frame_index][pre_index] = pre_similarity

                warp_matrix, _ = TrackUtil.ECC(self.all_image[pre_index], image)
                if warp_matrix is None:
                    iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                else:
                    aligned_boxes = TrackUtil.AffinePoints(self.all_boxes[pre_index], warp_matrix)
                    iou = TrackUtil.get_iou(aligned_boxes, boxes)
                self.all_iou[frame_index][pre_index] = iou

    def get_feature(self, frame_index, detection_index):
        '''
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes


class Node:
    '''
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    '''

    def __init__(self, frame_index, id):
        self.frame_index = frame_index
        self.id = id

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= TrackerConfig.max_record_frame:
            return None
        return recoder.all_boxes[self.frame_index][self.id, :]

    def get_iou(self, frame_index, recoder, box_id):
        if frame_index - self.frame_index >= TrackerConfig.max_track_node:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.id, box_id]


class Track:
    '''
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    '''
    _id_pool = 0

    def __init__(self):
        self.nodes = list()
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.valid = True  # indicate this track is merged
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        # iou judge
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.id)
            delta_frame = frame_index - n.frame_index
            if delta_frame in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_frame)
                # if iou < TrackerConfig.min_iou[iou_index]:
                if iou < TrackerConfig.min_iou[-1]:
                    return False
        self.nodes.append(node)
        self.reset_age()
        return True

    def get_similarity(self, frame_index, recorder):
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.id
            if frame_index - f >= TrackerConfig.max_track_node:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]

        if len(similarity) == 0:
            return None
        a = np.array(similarity)
        return np.sum(np.array(similarity), axis=0)

    def verify(self, frame_index, recorder, box_id):
        for n in self.nodes:
            delta_f = frame_index - n.frame_index
            if delta_f in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_f)
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou is None:
                    continue
                if iou < TrackerConfig.min_iou[iou_index]:
                    return False
        return True


class Tracks:
    '''
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''

    def __init__(self):
        self.tracks = list()  # the set of tracks
        self.max_drawing_track = TrackerConfig.max_draw_track_node

    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        self.tracks.append(track)
        self.volatile_tracks()

    def volatile_tracks(self):
        if len(self.tracks) > TrackerConfig.max_object:
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        ids = []
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder)
            if s is None:
                continue
            similarity += [s]
            ids += [t.id]

        similarity = np.array(similarity)

        track_num = similarity.shape[0]
        if track_num > 0:
            box_num = similarity.shape[1]
        else:
            box_num = 0

        if track_num == 0:
            return np.array(similarity), np.array(ids)

        similarity = np.repeat(similarity, [1] * (box_num - 1) + [track_num], axis=1)
        return np.array(similarity), np.array(ids)

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age > TrackerConfig.max_track_age:
                continue
            keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def merge(self, frame_index, recorder):
        t_l = len(self.tracks)
        res = np.zeros((t_l, t_l), dtype=float)
        # get track similarity matrix
        for i, t1 in enumerate(self.tracks):
            for j, t2 in enumerate(self.tracks):
                s = TrackUtil.get_merge_similarity(t1, t2, frame_index, recorder)
                if s is None:
                    res[i, j] = 0
                else:
                    res[i, j] = s

        # get the track pair which needs merged
        used_indexes = []
        merge_pair = []
        for i, t1 in enumerate(self.tracks):
            if i in used_indexes:
                continue
            max_track_index = np.argmax(res[i, :])
            if i != max_track_index and res[i, max_track_index] > TrackerConfig.min_merge_threshold:
                used_indexes += [max_track_index]
                merge_pair += [(i, max_track_index)]

        # start merge
        for i, j in merge_pair:
            TrackUtil.merge(self.tracks[i], self.tracks[j])

        # remove the invalid tracks
        self.tracks = [t for t in self.tracks if t.valid]

    def show(self, frame_index, recorder, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age < 2:
                b = t.nodes[-1].get_box(frame_index, recorder)
                if b is None:
                    continue
                txt = '({}, {})'.format(t.id, t.nodes[-1].id)
                image = cv2.putText(image, txt, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color,
                                    3)
                image = cv2.rectangle(image, (int(b[0]), int(b[1])),
                                      (int((b[2])), int(b[3])), t.color, 2)

        return image


# The tracker is compatible with pytorch (cuda)
class SSTTracker:
    def __init__(self):
        Track._id_pool = 0
        self.first_run = True
        self.image_size = TrackerConfig.image_size
        self.model_path = TrackerConfig.sst_model_path
        self.cuda = TrackerConfig.cuda
        self.mean_pixel = TrackerConfig.mean_pixel
        self.max_object = TrackerConfig.max_object
        self.frame_index = 0
        self.load_model()
        self.recorder = FeatureRecorder()
        self.tracks = Tracks()

    def load_model(self):
        # load the model
        self.model = EfficientNet(0)
        if self.cuda:
            cudnn.benchmark = True
            self.model.load_state_dict(torch.load(config['resume']))
            self.model = self.model.type(torch.DoubleTensor).cuda()
        else:
            self.model.load_state_dict(torch.load(config['resume'], map_location='cpu'))
            self.model = self.model.type(torch.DoubleTensor)
        self.model.eval()

    def update(self, image, detection, show_image, frame_index, force_init=False):
        '''
        Update the state of tracker, the following jobs should be done:
        1) extract the features
        2) stack the features together
        3) get the similarity matrix
        4) do assignment work
        5) save the previous image
        :param image: the opencv readed image, format is hxwx3
        :param detections: detection array. numpy array (l, r, w, h) and they all formated in (0, 1)
        '''

        self.frame_index = frame_index

        # format the image and detection
        detection_org = np.copy(detection)
        detection = TrackUtil.convert_detection(detection)
        detection_org[:, 2:] = detection_org[:, 2:] + detection_org[:, :2]
        h, w, _ = image.shape
        image_org = np.copy(image)
        image = TrackUtil.convert_image(image, detection)


        # features can be (1, 10, 450)
        features = self.model.forward_once(image)

        # update recorder
        self.recorder.update(self.frame_index, features.data, detection_org, image_org)

        if self.frame_index == 0 or force_init or len(self.tracks.tracks) == 0:
            for i in range(detection.shape[1]):
                t = Track()
                n = Node(self.frame_index, i)
                t.add_node(self.frame_index, self.recorder, n)
                self.tracks.append(t)
            self.tracks.one_frame_pass()
            # self.frame_index += 1
            return self.tracks.show(self.frame_index, self.recorder, image_org)

        # get tracks similarity
        y, ids = self.tracks.get_similarity(self.frame_index, self.recorder)

        if len(y) > 0:
            # 3) find the corresponding by the similar matrix
            """
            # 匈牙利算法，输出行列索引
            求 [[4,1,3],[2,0,5],[3,2,2]] 的最小开销
            行 [0 1 2] ， 列 [1 0 2]， 选取的元素 [1 2 2]
            """
            row_index, col_index = linear_sum_assignment(-y)
            col_index[col_index >= detection_org.shape[0]] = -1

            # verification by iou
            verify_iteration = 0
            # 最大验证次数是6次
            while verify_iteration < TrackerConfig.roi_verify_max_iteration:
                is_change_y = False
                for i in row_index:
                    box_id = col_index[i]
                    track_id = ids[i]

                    if box_id < 0:
                        continue
                    t = self.tracks.get_track_by_id(track_id)
                    if not t.verify(self.frame_index, self.recorder, box_id):
                        y[i, box_id] *= TrackerConfig.roi_verify_punish_rate
                        is_change_y = True
                if is_change_y:
                    row_index, col_index = linear_sum_assignment(-y)
                    col_index[col_index >= detection_org.shape[0]] = -1
                else:
                    break
                verify_iteration += 1

            # 4) update the tracks
            for i in row_index:
                track_id = ids[i]
                t = self.tracks.get_track_by_id(track_id)
                col_id = col_index[i]
                if col_id < 0:
                    continue
                node = Node(self.frame_index, col_id)
                t.add_node(self.frame_index, self.recorder, node)

            # 5) add new track
            for col in range(len(detection_org)):
                if col not in col_index:
                    node = Node(self.frame_index, col)
                    t = Track()
                    t.add_node(self.frame_index, self.recorder, node)
                    self.tracks.append(t)

        # remove the old track
        self.tracks.one_frame_pass()

        # merge the tracks
        # if self.frame_index % 20 == 0:
        #     self.tracks.merge(self.frame_index, self.recorder)

        # if show_image:
        image_org = self.tracks.show(self.frame_index, self.recorder, image_org)
        # self.frame_index += 1
        return image_org

        # self.frame_index += 1
        # image_org = cv2.resize(image_org, (320, 240))
        # vw.write(image_org)

        # plt.imshow(image_org)
