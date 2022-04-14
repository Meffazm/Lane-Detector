from os.path import join, exists
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
import cv2


class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, size=(256, 512)):
        self.dataset_path = dataset_path
        self.train = train
        self.size = size
        self.image_list = []
        self.lanes_list = []

        assert exists(self.dataset_path), f'Directory {self.dataset_path} does not exist!'

        label_files = []
        if self.train:
            label_files.append(join(self.dataset_path, 'label_data_0313.json'))
            label_files.append(join(self.dataset_path, 'label_data_0531.json'))
        else:
            label_files.append(join(self.dataset_path, 'label_data_0601.json'))

        for file in label_files:
            try:
                for line in open(file).readlines():
                    info_dict = json.loads(line)
                    self.image_list.append(info_dict['raw_file'])
                    h_samples = info_dict['h_samples']
                    lanes = info_dict['lanes']
                    xy_list = []

                    for lane in lanes:
                        y = np.array([h_samples]).T
                        x = np.array([lane]).T
                        xy = np.hstack((x, y))
                        index = np.where(xy[:, 0] > 2)
                        xy_list.append(xy[index])
                    self.lanes_list.append(xy_list)
            except BaseException:
                raise Exception(f'Fail to load {file}')

    def __getitem__(self, idx):

        img_path = join(self.dataset_path, self.image_list[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255

        segmentation_image = np.zeros((h, w))
        instance_image = np.zeros((h, w))

        lanes = self.lanes_list[idx]
        for idx, lane in enumerate(lanes):
            cv2.polylines(segmentation_image, [lane], False, 1, 10)
            cv2.polylines(instance_image, [lane], False, idx + 1, 10)

        segmentation_image = cv2.resize(segmentation_image, self.size, interpolation=cv2.INTER_NEAREST)
        instance_image = cv2.resize(instance_image, self.size, interpolation=cv2.INTER_NEAREST)

        segmentation_image = torch.from_numpy(segmentation_image).long()
        instance_image = torch.from_numpy(instance_image).long()

        # augmentations
        if self.train:
            tr_color = tt.Compose([tt.ToPILImage(),
                                   tt.ColorJitter(brightness=.3, hue=.3, saturation=.3),
                                   tt.ToTensor()])
            image = tr_color(image)

        return image, segmentation_image, instance_image  # 1 x H x W [[0, 1], [2, 0]]

    def __len__(self):
        return len(self.image_list)
