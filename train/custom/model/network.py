import torch
import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(
        self, 
        backbone, 
        rpn, 
        roi_head, 
        ):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_head
        self.initialize_weights()

    @torch.jit.ignore
    def forward_train(self, image, target):
        feat = self.backbone(image)
        rpn_output = self.rpn(image, feat)
        loss_rpn = self.rpn.get_loss(rpn_output, target)
        loss_roi = self.roi_heads.get_loss(feat, rpn_output, target)
        loss = dict()
        loss.update(loss_rpn)
        loss.update(loss_roi)
        return loss


    @torch.jit.export
    def forward(self, image):
        feat = self.backbone(image)
        rpn_output = self.rpn(image, feat)
        pred_boxes, pred_labels, pred_scores = self.roi_heads(image, feat, rpn_output)
        return pred_boxes, pred_labels, pred_scores


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
