import cv2
import os
import torch
import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils import data
from sklearn.model_selection import train_test_split


class KaggleCar(data.Dataset):
    def __init__(self, root, phase='train', transform=None, split=0.2, grid_size=7, num_bboxes=2, num_classes=20):
        self.image_root = os.path.join(root, 'training_images')
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

        if transform is None:
            # Normalize image and convert to tensor.
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', min_area=1024))
        else:
            self.transform = transform

        annot_file = os.path.join(root, 'train_solution_bounding_boxes.csv')
        samples = []
        df = pd.read_csv(annot_file)
        bboxes = {}
        for i, row in df.iterrows():
            im_name = row["image"]
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

            x_center = (xmax + xmin) / 2
            y_center = (ymax + ymin) / 2
            w = xmax - xmin
            h = ymax - ymin

            img = cv2.imread(os.path.join(self.image_root, im_name))
            h_img, w_img = img.shape[:2]

            x_center = x_center / w_img
            y_center = y_center / h_img
            w = w / w_img
            h = h / h_img

            if im_name not in bboxes:
                bboxes[im_name] = [np.asarray([x_center, y_center, w, h, 0])]
            else:
                bboxes[im_name].append(np.asarray([x_center, y_center, w, h, 0]))

        for k, v in bboxes.items():
            samples.append((k, v))

        if phase == 'train':
            self.samples, _ = train_test_split(samples, test_size=split, random_state=42)
        else:
            _, self.samples = train_test_split(samples, test_size=split, random_state=42)

    def _encode_bbox(self, bboxes):
        boxes = torch.tensor([bbox[:4] for bbox in bboxes])
        labels = torch.tensor([int(bbox[4]) for bbox in bboxes])

        N = 5 * self.B + self.C
        target = torch.zeros(self.S, self.S, N)
        cell_size = 1.0 / float(self.S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], labels[b]

            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size  # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0.

            for k in range(self.B):
                s = 5 * k
                target[j, i, s:s + 2] = xy_normalized
                target[j, i, s + 2:s + 4] = wh
                target[j, i, s + 4] = 1.0
            target[j, i, 5 * self.B + label] = 1.0

        return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        im_name, bboxes = self.samples[index]
        image = cv2.imread(os.path.join(self.image_root, im_name))

        if self.transform is not None:
            result = self.transform(image=image, bboxes=bboxes)
            image = result["image"]
            bboxes = result["bboxes"]
        bboxes = self._encode_bbox(bboxes)

        return image, bboxes


def kaggle_car_loader_factory_v1(root, img_h=448, img_w=448, bs=32):
    transform = A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', min_area=1024))

    train_ds = KaggleCar(root=root, phase='train', num_classes=1, transform=transform)
    train_loader = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)

    val_ds = KaggleCar(root=root, phase='val', num_classes=1, transform=transform)
    val_loader = data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    return train_loader, val_loader
