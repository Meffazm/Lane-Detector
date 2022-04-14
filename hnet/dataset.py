import numpy as np
from os.path import join, exists
import json
import torch
from torch.utils.data import Dataset
import cv2


class HomographyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, train=True, size=(64, 128)):
        self.dataset_path = dataset_path
        self.train = train
        self.size = size
        self.image_list = []
        self.lanes_list = []
        self.max_lanes = 0
        self.max_points = 0

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
                    self.max_lanes = max(self.max_lanes, len(lanes))
                    xy_list = []

                    for lane in lanes:
                        y = np.array([h_samples]).T
                        x = np.array([lane]).T
                        xy = np.hstack((x, y))
                        index = np.where(xy[:, 0] > 2)
                        xy_list.append(xy[index])
                        self.max_points = max(self.max_points, len(xy[index]))
                    self.lanes_list.append(xy_list)
            except BaseException:
                raise Exception(f'Fail to load {file}')

    def __getitem__(self, idx):
        img_path = join(self.dataset_path, self.image_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255

        # creating output with shape [max_lanes, 2, max_points]
        buffer = None

        for lane in self.lanes_list[idx]:
            lane = np.expand_dims(np.pad(np.swapaxes(lane, 0, 1),
                                         pad_width=((0, 0), (0, self.max_points - lane.shape[0])),
                                         mode='constant',
                                         constant_values=0), 0)

            if buffer is not None:
                buffer = np.concatenate((buffer, lane), 0)
            else:
                buffer = lane

        ground_truth_trajectory = torch.from_numpy(np.pad(buffer,
                                                          pad_width=((0, self.max_lanes - buffer.shape[0]),
                                                                     (0, 0),
                                                                     (0, 0)),
                                                          mode='constant',
                                                          constant_values=0))

        return image, ground_truth_trajectory

    def __len__(self):
        return len(self.image_list)
