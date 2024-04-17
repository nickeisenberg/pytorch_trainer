import numpy as np
from collections import defaultdict
from collections.abc import Callable
import os
import json
from torch import Tensor, tensor
from PIL import Image
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .targets import build_yolo_target


def yolo_unpacker(data, device):
    img_ten = data[0].to(device)
    yolo_target_tuple = tuple([d.to(device) for d in data[1]])
    return img_ten, yolo_target_tuple


def make_data_from_coco(coco: dict, file_name_root: str, boxes_as_tensors=True) -> dict:
    """
    {
        idx: {
            "file_name": ...,
            "image_id": ...,
            "bbox": [...],
            "category_id": [...]
        }
    }
    """
    
    data = defaultdict(dict)
    img_id_to_idx = {}
    for img in coco["images"]:
        img_id = img["id"]
        file_name = os.path.join(file_name_root, img["file_name"])

        idx = len(data)
        data[idx] = {
            "file_name": file_name,
            "image_id": img_id,
            "bboxes": [],
            "category_ids": [] 
        }
        img_id_to_idx[img_id] = idx
    
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        idx = img_id_to_idx[img_id]
        if boxes_as_tensors:
            data[idx]["bboxes"].append(tensor(ann["bbox"]))
        else:
            data[idx]["bboxes"].append(ann["bbox"])
        data[idx]["category_ids"].append(ann["category_id"])

    return data


class YoloDataset(Dataset):
    def __init__(self, 
                 coco: str | dict,
                 img_root: str,
                 return_shape: tuple[tuple, ...],
                 normalized_anchors: Tensor,
                 scales: list[int],
                 iou_thresh: float = 0.5,
                 img_tranform: Callable = Compose([Image.open, ToTensor()])):

        if isinstance(coco, str):
            with open(coco, "r") as oj:
                self.coco = json.load(oj)
        else:
            self.coco = coco

        self.img_root = img_root
        
        self.return_shape = return_shape
        self.normalized_anchors = normalized_anchors
        self.scales = scales
        self.iou_thresh = iou_thresh

        self.img_transform = img_tranform

        self.data = make_data_from_coco(self.coco, self.img_root)


    def __getitem__(self, idx):
        data = self.data[idx]
        file_name = data["file_name"]
        bboxes = data["bboxes"]
        category_ids = data["category_ids"]

        input = self.img_transform(file_name)

        target = build_yolo_target(
            return_shape=self.return_shape,
            bboxes=bboxes,
            label_ids=category_ids,
            normalized_anchors=self.normalized_anchors,
            scales=self.scales,
            img_width=input.shape[-1],
            img_height=input.shape[-2],
            iou_thresh=self.iou_thresh
        )

        return input, target


    def __len__(self):
        return len(self.data)


class YoloDataset2(Dataset):
    def __init__(self, 
                 coco: str | dict,
                 img_root: str,
                 return_shape: tuple[tuple, ...],
                 normalized_anchors: Tensor,
                 scales: list[int],
                 iou_thresh: float = 0.5,
                 transform: Callable | None = None):

        if isinstance(coco, str):
            with open(coco, "r") as oj:
                self.coco = json.load(oj)
        else:
            self.coco = coco

        self.img_root = img_root
        
        self.return_shape = return_shape
        self.normalized_anchors = normalized_anchors
        self.scales = scales
        self.iou_thresh = iou_thresh
        
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=[0], std=[1], max_pixel_value=255,),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(
                    format="coco", label_fields=["labels"],
                ),
            )

        self.data = make_data_from_coco(self.coco, self.img_root, False)


    def __getitem__(self, idx):
        data = self.data[idx]
        file_name = data["file_name"]
        boxes = data["bboxes"]
        labels = data["category_ids"]

        image = np.array(Image.open(file_name).convert("L"))

        augmentations = self.transform(
            image=image, bboxes=boxes, labels=labels
        )

        image = augmentations["image"]
        boxes = [tensor(x) for x in augmentations["bboxes"]]
        labels= augmentations["labels"]

        target = build_yolo_target(
            return_shape=self.return_shape,
            bboxes=boxes,
            label_ids=labels,
            normalized_anchors=self.normalized_anchors,
            scales=self.scales,
            img_width=image.shape[-1],
            img_height=image.shape[-2],
            iou_thresh=self.iou_thresh
        )

        return image, target


    def __len__(self):
        return len(self.data)
