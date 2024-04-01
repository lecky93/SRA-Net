import os

import torch
import torch.utils.data as data
import numpy as np
import cv2
import copy
from mmengine.registry import DATASETS
from mmengine.fileio import list_from_file, load
from mmengine.logging import print_log
import logging

from .transform import build_transform
from .segment_data import SegData

@DATASETS.register_module()
class CasiaDataset(data.Dataset):
    METAINFO = dict(
        classes=('real', 'fake'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 data_root,
                 img_dir=None,
                 ann_dir=None,
                 ann_suffix='',
                 ann_file=None,
                 is_train=True,
                 split=None,
                 img_size=(512, 512),
                 metainfo=None,
                 ):
        super(CasiaDataset, self).__init__()
        self.data_path = data_root
        self.is_train = is_train
        if img_dir is not None:
            self.img_dir = os.path.join(data_root, img_dir)
        if ann_dir is not None:
            self.ann_dir = os.path.join(data_root, ann_dir)

        self.transform = build_transform(is_train, img_size=img_size)

        self.database = []
        self.basename = []
        if split is not None:
            file = open(os.path.join(data_root, split))
            lines = file.readlines()
            for line in lines:
                line = line.replace('\n', '')
                self.basename.append(line)
                img_path = os.path.join(self.img_dir, line + '.jpg')
                ann_path = os.path.join(self.ann_dir, line + '_gt.png')
                self.database.append([img_path, ann_path, 1])
            file.close()

        if ann_file is not None:
            file = open(os.path.join(self.data_path, ann_file))
            lines = file.readlines()
            for line in lines:
                tmp = line.split(',')
                self.basename.append(tmp[0].split('/')[-1].split('.')[0])
                if tmp[1] != 'None':
                    self.database.append([os.path.join(self.data_path, tmp[0]), os.path.join(self.data_path, tmp[1]), int(tmp[2])])
                else:
                    self.database.append([os.path.join(self.data_path, tmp[0]), tmp[1], int(tmp[2])])
            file.close()

        else:
            img_files = os.listdir(self.img_dir)
            for file in img_files:
                if file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    file_name = file.split('.')[0]
                    self.basename.append(file_name)
                    ann_name = file_name + ann_suffix + '.png'
                    img = os.path.join(self.img_dir, file)
                    ann = os.path.join(self.ann_dir, ann_name)
                    self.database.append([img, ann, 1])

        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Meta information dict. If ``metainfo``
                contains existed filename, it will be parsed by
                ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def __getitem__(self, index):
        idb = self.database[index]
        name = self.basename[index]
        img = cv2.imread(idb[0])

        is_splicing = False
        # if self.is_train:
        #     is_splicing = idb[0].split('/')[-1].split('_')[1] == 'D'

        size = img.shape[:2]
        label = np.zeros(img.shape)
        if idb[1] != 'None':
            label = cv2.imread(idb[1])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {'image': img, 'mask': label}

        if self.transform is not None:
            sample = self.transform(image=np.array(img), mask=np.array(label))
        img = sample['image']
        label = sample['mask']

        label_size = label.shape[:2]
        label = np.transpose(label, (2, 0, 1))
        label = label[0, :, :]

        # target
        target = torch.Tensor([int(idb[2])])

        # data_sample = dict()
        # data_sample['img_path'] = idb[0]
        # data_sample['ori_size'] = str(size)
        # data_sample['gt_label'] = label
        # data_sample['cls_lebel'] = target

        data_samples = SegData(
            img_path=idb[0],
            img_name=name,
            ori_size=size,
            label_size=label_size,
            gt_label=label,
            cls_label=target,
            is_splicing=is_splicing
        )

        return img, data_samples

    def __len__(self):
        return len(self.database)