import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import MeanShift
import torch


class LaneDetector:

    def __init__(self, model, cluster_method="MeanShift"):
        self.model = model
        self.cluster_method = cluster_method

    def detect(self, img, y_positions=None):

        binary_final_logits, instance_embedding = self.model(img.unsqueeze(0))
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        rgb_emb, cluster_result = self.process_instance_embedding(instance_embedding,
                                                                  binary_img,
                                                                  distance=1,
                                                                  lane_num=5,
                                                                  cluster_method=self.cluster_method)
        x_positions = []
        if y_positions is not None:
            for y in y_positions:
                line = binary_img[y].astype(int)
                x_step = []
                x_positions.append(x_step)
            for x in range(1, len(line)):
                if line[x - 1] != line[x] and line[x] == 0:
                    x_step.append(x)

        return binary_final_logits, instance_embedding, binary_img, rgb_emb, cluster_result, x_positions

    @staticmethod
    def process_instance_embedding(instance_embedding, binary_img, distance, lane_num, cluster_method):
        embedding = instance_embedding[0].detach().numpy().transpose(1, 2, 0)
        cluster_result = np.zeros(binary_img.shape, dtype=np.int32)
        cluster_list = embedding[binary_img > 0]

        if cluster_method == 'MeanShift':
            mean_shift = MeanShift(bandwidth=distance, bin_seeding=True, n_jobs=-1)
            mean_shift.fit(cluster_list)
            labels = mean_shift.labels_
        elif cluster_method == 'HDBSCAN':
            hdb = HDBSCAN()
            hdb.fit(cluster_list)
            labels = hdb.labels_

        cluster_result[binary_img > 0] = labels + 1
        cluster_result[cluster_result > lane_num] = 0

        for idx in np.unique(cluster_result):
            if len(cluster_result[cluster_result == idx]) < 15:
                cluster_result[cluster_result == idx] = 0

        H, W = binary_img.shape
        rgb_emb = np.zeros((H, W, 3))
        color = [[0, 0, 0],
                 [255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [255, 215, 0],
                 [0, 255, 255]]
        element = np.unique(cluster_result)

        for i in range(len(element)):
            rgb_emb[cluster_result == element[i]] = color[i]

        return rgb_emb / 255, cluster_result
