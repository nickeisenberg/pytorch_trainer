def map_evaluate(self,
                 loader: DataLoader, 
                 unpacker: Callable,
                 min_box_dim: tuple[int, int],
                 score_thresh=.95,
                 nms_iou_thresh=.3):

    self.model.eval()

    map = MeanAveragePrecision(
        box_format='xywh'
    )
    map.warn_on_many_detections = False

    pbar = tqdm(loader)
    for data in pbar:
        inputs, targets = unpacker(data, self.device)

        assert type(inputs) == Tensor
        targets = cast(tuple[Tensor, ...], targets)
        
        with no_grad():
            predictions = self.model(inputs)
            decoded_predictions = decode_yolo_tuple(
                yolo_tuple=predictions, 
                img_width=self.img_width, 
                img_height=self.img_height, 
                normalized_anchors=self.normalized_anchors, 
                scales=self.scales, 
                score_thresh=score_thresh,
                nms_iou_thresh=nms_iou_thresh,
                min_box_dim=min_box_dim,
                is_pred=True
            )
            decoded_targets = decode_yolo_tuple(
                yolo_tuple=targets, 
                img_width=self.img_width, 
                img_height=self.img_height, 
                normalized_anchors=self.normalized_anchors, 
                scales=self.scales, 
                is_pred=False
            )
            map.update(preds=decoded_predictions, target=decoded_targets)

        computes = map.compute()
        pbar.set_postfix(
                map=computes["map"],
                map_50=computes["map_50"],
                map_75=computes["map_75"]
        )

    return map

