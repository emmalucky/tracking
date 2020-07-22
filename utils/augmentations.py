import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from config.config import config
import utils.operation as op


# finish: add the random rectangle
# finish: add the random gap
# finish: add false object label
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        for t in self.transforms:
            img_pre, img_next, boxes_pre, boxes_next, labels = \
                t(img_pre, img_next, boxes_pre, boxes_next, labels)
        return img_pre, img_next, boxes_pre, boxes_next, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        img_pre = img_pre.astype(np.float32)
        img_pre -= self.mean
        img_next = img_next.astype(np.float32)
        img_next -= self.mean
        return img_pre.astype(np.float32), img_next.astype(np.float32), boxes_pre, boxes_next, labels


class ResizeShuffleBoxes(object):
    def show_matching_hanlded_rectangle(self, img_pre, img_next, boxes_pre, boxes_next, labels):
        img_pre = (img_pre + np.array(config['mean_pixel'])).astype(np.uint8)
        img_next = (img_next + np.array(config['mean_pixel'])).astype(np.uint8)
        h = img_pre.shape[0]

        return op.show_matching_rectangle(img_pre, img_next, boxes_pre[:, :] * h, boxes_next[:, :] * h, labels)

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        resize_f = lambda boxes: \
            (boxes.shape[0],
             np.vstack((
                 boxes,
                 np.full(
                     (config['max_object'] - len(boxes),
                      boxes.shape[1]),
                     np.inf
                 ))))
        # show the shuffling result
        # cv2.imwrite('temp_handled.jpg', self.show_matching_hanlded_rectangle(img_pre, img_next, boxes_pre, boxes_next, labels))
        size_pre, boxes_pre = resize_f(boxes_pre)
        size_next, boxes_next = resize_f(boxes_next)

        indexes_pre = np.arange(config['max_object'])
        indexes_next = np.arange(config['max_object'])
        np.random.shuffle(indexes_pre)
        np.random.shuffle(indexes_next)

        boxes_pre = boxes_pre[indexes_pre, :]
        boxes_next = boxes_next[indexes_next, :]

        labels = labels[indexes_pre, :]
        labels = labels[:, indexes_next]

        mask_pre = indexes_pre < size_pre
        mask_next = indexes_next < size_next

        # add false object label
        false_object_pre = (labels.sum(1) == 0).astype(float)  # should consider unmatched object
        false_object_pre[np.logical_not(mask_pre)] = 0.0

        false_object_next = (labels.sum(0) == 0).astype(float)  # should consider unmatched object
        false_object_next[np.logical_not(mask_next)] = 0.0

        false_object_pre = np.expand_dims(false_object_pre, axis=1)
        labels = np.concatenate((labels, false_object_pre), axis=1)  # 60x61

        false_object_next = np.append(false_object_next, [0])
        false_object_next = np.expand_dims(false_object_next, axis=0)
        labels = np.concatenate((labels, false_object_next), axis=0)  # 60x61

        mask_pre = np.append(mask_pre, [True])  # 61
        mask_next = np.append(mask_next, [True])  # 61
        return img_pre, img_next, \
               [boxes_pre, mask_pre], \
               [boxes_next, mask_next], \
               labels


class FormatBoxes(object):
    '''
    note: format the label in order to input into the selector net.
    '''

    def __init__(self, keep_box=False, delta=config['crop_delta']):
        self.keep_box = keep_box
        self.delta = delta

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        '''
        boxes_pre: [N, 4]
        '''
        if not self.keep_box:
            # convert the center to [-1, 1]
            f = lambda boxes: np.concatenate([boxes[:, :2] + boxes[:, 2:] / 2, boxes[:, 2:] * self.delta], axis=1)
        else:
            f = lambda boxes: np.expand_dims(
                np.expand_dims(
                    np.concatenate([(boxes[:, :2] + boxes[:, 2:]) - 1, boxes[:, 2:6]], axis=1),
                    axis=1
                ),
                axis=1
            )

        # remove inf
        boxes_pre[0] = f(boxes_pre[0])
        boxes_pre[0][boxes_pre[0] == np.inf] = 0

        boxes_next[0] = f(boxes_next[0])
        boxes_next[0][boxes_next[0] == np.inf] = 0

        return img_pre, img_next, boxes_pre, boxes_next, labels


class GetObject(object):
    def __init__(self, crop_size=config['crop_size']):
        self.size = crop_size

    def get_objects(self, image, boxes):
        # first version image shape 为 3*h*w，取错了
        h, w, _ = image.shape
        objects = np.zeros((1, self.size, self.size, 3))
        for box in boxes:
            # box = boxes[i]
            width = int(box[2])
            height = int(box[3])
            if width == 0 or height == 0:
                ob = np.zeros((1, self.size, self.size, 3))
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
                ob = cv2.resize(ob, (self.size, self.size))
                ob = np.expand_dims(ob, axis=0)
            objects = np.concatenate([objects, ob], 0)
        return objects[1:]

    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        img_pre = self.get_objects(img_pre, boxes_pre[0])
        img_next = self.get_objects(img_next, boxes_next[0])
        return img_pre, img_next, boxes_pre, boxes_next, labels


class ToTensor(object):
    def __call__(self, img_pre, img_next, boxes_pre=None, boxes_next=None, labels=None):
        img_pre = torch.from_numpy(img_pre.astype(np.float32)).permute(0, 3, 1, 2)
        img_next = torch.from_numpy(img_next.astype(np.float32)).permute(0, 3, 1, 2)

        boxes_pre[0] = torch.from_numpy(boxes_pre[0].astype(float))
        boxes_pre[1] = torch.from_numpy(boxes_pre[1].astype(np.uint8))

        boxes_next[0] = torch.from_numpy(boxes_next[0].astype(float))
        boxes_next[1] = torch.from_numpy(boxes_next[1].astype(np.uint8))

        labels = torch.from_numpy(labels).unsqueeze(0)

        return img_pre, img_next, boxes_pre, boxes_next, labels


class SSJAugmentation(object):
    def __init__(self, size=config['sst_dim'], mean=config['mean_pixel'], type=config['type']):
        self.mean = mean
        self.size = size
        if type == 'train':
            self.augment = Compose([
                SubtractMeans(self.mean),
                ResizeShuffleBoxes(),
                FormatBoxes(),
                GetObject(),
                ToTensor()
            ])
        elif type == 'test':
            self.augment = Compose([
                SubtractMeans(self.mean),
                ResizeShuffleBoxes(),
                GetObject(),
                ToTensor()
            ])
        else:
            raise NameError('config type is wrong, should be choose from (train, test)')

    def __call__(self, img_pre, img_next, boxes_pre, boxes_next, labels):
        return self.augment(img_pre, img_next, boxes_pre, boxes_next, labels)


class SSJEvalAugment(object):
    def __init__(self, size=config['sst_dim'], mean=config['mean_pixel']):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            SubtractMeans(self.mean),
            ResizeShuffleBoxes(),
            GetObject(),
            ToTensor()
        ])

    def __call__(self, img_pre, img_next, boxes_pre, boxes_next, labels):
        return self.augment(img_pre, img_next, boxes_pre, boxes_next, labels)


def collate_fn(batch):
    img_pre = []
    img_next = []
    boxes_pre = []
    boxes_next = []
    labels = []
    indexes_pre = []
    indexes_next = []
    for sample in batch:
        img_pre.append(sample[0])
        img_next.append(sample[1])
        boxes_pre.append(sample[2][0].float())
        boxes_next.append(sample[3][0].float())
        labels.append(sample[4].float())
        indexes_pre.append(sample[2][1].byte())
        indexes_next.append(sample[3][1].byte())
    return torch.stack(img_pre, 0), torch.stack(img_next, 0), \
           torch.stack(boxes_pre, 0), torch.stack(boxes_next, 0), \
           torch.stack(labels, 0), \
           torch.stack(indexes_pre, 0).unsqueeze(1), \
           torch.stack(indexes_next, 0).unsqueeze(1)
