import torch
from enet.model import ENet
from enet.dataset import LaneDataset
from detector import LaneDetector


def main(data, enet_checkpoint, idx=0, cluster_method="MeanShift"):
    y_positions = [64, 127, 190]

    checkpoint = torch.load(enet_checkpoint)
    model = ENet()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    ld = LaneDetector(model=model, cluster_method=cluster_method)
    val_set = LaneDataset(data, train=False)
    img, seg_img, ins_img = val_set[idx]

    return ld.detect(img, y_positions)


if __name__ == "__main__":
    main(data='./tusimple/', enet_checkpoint='./checkpoints/enet.pth')
