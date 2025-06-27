from torchmetrics.classification import Recall, Precision, F1Score, Specificity
from torchmetrics.wrappers.abstract import WrapperMetric
import torchmetrics
from torch import Tensor


class SegmentationToClassificationWrapper(WrapperMetric):
    """
    Enables applying classification metrics to segmentation masks.
    The segmentation masks are converted to binary one-hot enconding vectors using a IoU threshold.

    Args:
        metric_cls: Segmentation metric class.
        iou_threshold: IoU threshold to convert segmentation to binary classification.
    """

    def __init__(self,
                 metric_cls: torchmetrics.Metric,
                 iou_threshold=0.5):
        super().__init__()
        self.metric = metric_cls
        self.iou_threshold = iou_threshold

    def update(self, preds: Tensor, target: Tensor):
        cls_pred, cls_target = self.transform_mask_to_binary(preds, target, self.iou_threshold)
        self.metric.update(cls_pred, cls_target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        super().reset()
        self.metric.reset()

    def forward(self, preds: Tensor, target: Tensor):
        cls_pred, cls_target = self.transform_mask_to_binary(preds, target, self.iou_threshold)
        return self.metric.forward(cls_pred, cls_target)

    @staticmethod
    def transform_mask_to_binary(pred: Tensor, target: Tensor, iou_threshold: float = 0.5) -> tuple[Tensor, Tensor]:
        """
        Convert both the segmentation masks with shape (B,L,H,W) to shape (B,L), so that classification metrics can be used.
        The conversion is done by applying using a IoU threshold that must be satisfied to consider a prediction as true positive.
        Args:
            pred: Segmentation prediction of shape (B, L, H, W).
            target: Segmentation target of shape (B, L, H, W).
            iou_threshold: IoU threshold to convert segmentation to binary classification.
        Returns:
            Tuple of binary classification predictions (first) and targets (second) of shape (B, L).
        """

        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError("Input tensors must have 4 dimensions (B, L, H, W).")

        # Calculate IoU for each sample
        intersection = (pred & target).float().sum(dim=(2, 3))
        union = (pred | target).float().sum(dim=(2, 3))
        iou = intersection / union

        cls_target = target.amax(dim=(2, 3))
        cls_pred = (iou >= iou_threshold).float()

        return cls_pred, cls_target

# # test
# cls_metric = Recall(task="multilabel", average="macro", num_labels=3)
# metric = SegmentationToClassificationWrapper(cls_metric, iou_threshold=0.33)
# metric(preds=torch.randint(0, 2, size=(4, 3, 32, 32)),
#        target=torch.randint(0, 2, size=(4, 3, 32, 32))
#        )