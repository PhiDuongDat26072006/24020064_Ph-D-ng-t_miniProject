import torch
import os
import pandas as pd
from PIL import Image
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, cvs_file, label_dir, image_dir, S = 7, C = 20, B = 2, transforms = None):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.S = S
        self.B = B
        self.C = C
        self.annotations = pd.read_csv(cvs_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_path)
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y , width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n', "").split()]
                boxes.append([class_label, x, y, width, height])

        boxes, image = self.transforms(boxes = boxes, image =  image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        boxes = torch.tensor(boxes)

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - i, self.S * y - j
            w_cell, h_cell = self.S * width, self.S * height

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, int(class_label)] = 1
                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C + 1 : self.C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])

        return image, label_matrix

class VOCDataset_org_data(VOCDataset):
    def __init__(self, cvs_file, label_dir, image_dir, S = 7, C = 20, B = 2, transforms = None):
        super().__init__(cvs_file, label_dir, image_dir, S, C, B, transforms)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_path)
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n', "").split()]
                boxes.append([class_label, x, y, width, height])

        boxes = torch.tensor(boxes)

        return image, boxes