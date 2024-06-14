from copy import deepcopy
from typing import cast
from torch.nn import Module
from torch import float32, no_grad, Tensor, tensor, vstack
from torch.optim import Adam


from trfc.callbacks import (
    CSVLogger,
    ProgressBarUpdater,
    SaveBestCheckoint
)

from ..yolov5 import YOLOv5
from ..yolo_utils.dataset import yolo_unpacker
from ..yolo_utils import YOLOLoss


class TrainModule(Module):
    def __init__(self,
                 yolo: YOLOv5,
                 device: int | str,
                 img_width: int,
                 img_height: int,
                 normalized_anchors: Tensor,
                 scales: list[int],
                 loss_log_root: str,
                 state_dict_root: str):
        super().__init__()

        self.device = device
        self.model = yolo.to(device)

        self.loss_fn = YOLOLoss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.img_width, self.img_height = img_width, img_height
        self.normalized_anchors = normalized_anchors
        self.scales = scales 
        
        _scaled_anchors = []
        for scale_id, scale in enumerate(self.scales):
            scaled_anchors = deepcopy(
                self.normalized_anchors[3 * scale_id: 3 * (scale_id + 1)].detach()
            )
            scaled_anchors *= tensor(
                [self.img_width / scale ,self.img_height / scale]
            )
            _scaled_anchors.append(scaled_anchors)

        self.scaled_anchors = vstack(_scaled_anchors).to(float32)
    
        self.logger = CSVLogger(loss_log_root)
        self.save_best_ckp = SaveBestCheckoint(
            state_dict_root, 
            key="total_loss"
        )
        self.pbar_updater = ProgressBarUpdater()


    def callbacks(self):
        return [self.logger, self.save_best_ckp, self.pbar_updater]


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, loader_data):
        self.model.train()

        inputs, targets = yolo_unpacker(loader_data, self.device)

        _device = inputs.device.type

        self.scaled_anchors = self.scaled_anchors.to(_device)

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        total_loss = tensor(0.0, requires_grad=True).to(_device)
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scaled_anchors = self.scaled_anchors[3 * scale_id: 3 * (scale_id + 1)]
            loss, batch_history = self.loss_fn(output, target, scaled_anchors)
            total_loss += loss
            self.logger.log(batch_history)

        total_loss.backward()

        self.optimizer.step()


    def validation_batch_pass(self, loader_data):
        self.model.eval()

        inputs, targets = yolo_unpacker(loader_data, self.device)

        _device = inputs.device.type

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)
        
        self.scaled_anchors = self.scaled_anchors.to(_device)
        
        with no_grad():
            outputs = self.model(inputs)
        
        for scale_id, (output, target) in enumerate(zip(outputs, targets)):
            scaled_anchors = self.scaled_anchors[3 * scale_id: 3 * (scale_id + 1)]
            _, batch_history = self.loss_fn(output, target, scaled_anchors)
            self.logger.log(batch_history)
