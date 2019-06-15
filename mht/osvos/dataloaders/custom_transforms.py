import torch
import math
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': img,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(sample['label']).astype(np.float32), -1).transpose((2, 0, 1))
        mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()


        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        # self.size = tuple(reversed(size))  # size: (h, w)
        self.size = (int(size), int(size))

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # assert img.size == mask.size

        img = Image.fromarray(np.uint8(img)).resize(self.size, Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        # img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(img.shape)
        # print(mask.shape)
        assert img.size == mask.size
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.size == mask.size
        # print(img.shape[0])
        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])
        # print(w,h)
        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}

        return self.crop(self.scale(sample))

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'fname' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)
            # print(elem)

            sample[elem] = tmp

        return sample


# class Resize(object):
#     """Randomly resize the image and the ground truth to specified scales.
#     Args:
#         scales (list): the list of scales
#     """
#     def __init__(self, scales=[0.5, 0.8, 1]):
#         self.scales = scales

#     def __call__(self, sample):

#         # Fixed range of scales
#         sc = self.scales[random.randint(0, len(self.scales) - 1)]

#         for elem in sample.keys():
#             if 'fname' in elem:
#                 continue
#             tmp = sample[elem]

#             if tmp.ndim == 2:
#                 flagval = cv2.INTER_NEAREST
#             else:
#                 flagval = cv2.INTER_CUBIC

#             tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

#             sample[elem] = tmp

#         return sample

class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.size

        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            # print(tmp.shape)
            tmp = cv2.resize(tmp, (sc,sc), interpolation=flagval)
            # print(tmp.shape)

            sample[elem] = tmp

        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'fname' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(np.array(tmp).astype('float32'), flipCode=1)
                sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # print(tmp.shape)
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample
