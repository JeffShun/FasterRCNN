import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, batched_nms
from torchvision.ops import roi_align

from .utils import (
    get_iou, 
    boxes_to_transformation_targets, 
    apply_regression_pred_to_anchors_or_proposals,
    clamp_boxes_to_image_boundary,
    sample_positive_negative
    )

class ROIHead(nn.Module):
    r"""
    ROI head on top of ROI pooling layer for generating
    classification and box transformation predictions
    We have two fc layers followed by a classification fc layer
    and a bbox regression fc layer
    """
    def __init__(
        self, 
        in_channels,
        num_classes,
        spatial_scale,
        roi_iou_threshold,
        roi_low_bg_iou,
        roi_nms_threshold,
        roi_min_boxsize,
        roi_topk_detections,
        roi_score_threshold,
        roi_pool_size,
        fc_inner_dim,
        roi_loss_count,
        roi_pos_fraction
        ):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.spatial_scale = spatial_scale
        self.iou_threshold = roi_iou_threshold
        self.low_bg_iou = roi_low_bg_iou
        self.nms_threshold = roi_nms_threshold
        self.min_boxsize = roi_min_boxsize
        self.topK_detections = roi_topk_detections
        self.low_score_threshold = roi_score_threshold
        self.pool_size = roi_pool_size
        self.fc_inner_dim = fc_inner_dim
        self.roi_loss_count = roi_loss_count
        self.roi_pos_count = int(roi_pos_fraction* self.roi_loss_count)
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)


    def forward(self, image, feat, rpn_output): 
        # Get desired scale to pass to roi pooling function
        proposals = rpn_output['proposals']
        image_shape = image.shape[-2:]
        proposal_roi_pool_feats = roi_align(feat, [proposals], output_size=self.pool_size, spatial_scale=self.spatial_scale, sampling_ratio=2, aligned=True)
        
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = F.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = F.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        # cls_scores -> (proposals, num_classes)
        # box_transform_pred -> (proposals, num_classes * 4)
        ##############################################
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        
        device = cls_scores.device
        # Apply transformation predictions to proposals
        pred_boxes = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
        pred_scores = F.softmax(cls_scores, dim=-1)
        
        # Clamp box to image boundary
        pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)
        
        # create labels for each prediction
        pred_labels = torch.arange(num_classes, device=device)
        pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

        # remove predictions with the background label
        pred_boxes = pred_boxes[:, 1:]
        pred_scores = pred_scores[:, 1:]
        pred_labels = pred_labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        pred_boxes = pred_boxes.reshape(-1, 4)
        pred_scores = pred_scores.reshape(-1)
        pred_labels = pred_labels.reshape(-1) - 1  # remove background label
        
        pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)   

        return pred_boxes, pred_labels, pred_scores
    
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        r"""
        Given a set of proposals and ground truth boxes and their respective labels.
        Use IOU to assign these proposals to some gt box or background
        :param proposals: (number_of_proposals, 4)
        :param gt_boxes: (number_of_gt_boxes, 4)
        :param gt_labels: (number_of_gt_boxes)
        :return:
            labels: (number_of_proposals)
            matched_gt_boxes: (number_of_proposals, 4)
        """
        # Get IOU Matrix between gt boxes and proposals
        iou_matrix = get_iou(gt_boxes, proposals)
        # For each gt box proposal find best matching gt box
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < self.iou_threshold) & (best_match_iou >= self.low_bg_iou)
        ignored_proposals = best_match_iou < self.low_bg_iou
        
        # Update best match of low IOU proposals to -1
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2
        
        # Get best marching gt boxes for ALL proposals
        # Even background proposals would have a gt box assigned to it
        # Label will be used to ignore them later
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Get class label for all proposals according to matching gt boxes
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        # Update background proposals to be of label 0(background)
        labels[background_proposals] = 0
        
        # Set all to be ignored anchor labels as -1(will be ignored)
        labels[ignored_proposals] = -1
        
        return labels, matched_gt_boxes_for_proposals

    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        r"""
        Method to filter predictions by applying the following in order:
        1. Filter low scoring boxes
        2. Remove small size boxes
        3. NMS for each class separately
        4. Keep only topK detections
        :param pred_boxes:
        :param pred_labels:
        :param pred_scores:
        :return:
        """
        # 1. remove low scoring boxes
        keep = torch.where(pred_scores > self.low_score_threshold)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # 2. Remove small boxes
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= self.min_boxsize) & (hs >= self.min_boxsize)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        # 3. Class wise NMS（用 batched_nms 一步搞定）
        if pred_boxes.numel() == 0:
            return pred_boxes, pred_labels, pred_scores

        keep = batched_nms(
            pred_boxes,
            pred_scores,
            pred_labels,
            self.nms_threshold
            )

        # 4. Keep only topK
        keep = keep[:self.topK_detections]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

        return pred_boxes, pred_labels, pred_scores
    

    def get_loss(self, feat, rpn_output, target):
        proposals = rpn_output['proposals']
        target = target[0]
        target = target[target[:, 4] != -1]
        # Add ground truth to proposals
        proposals = torch.cat([proposals, target[:,:4]], dim=0)
        
        gt_boxes = target[:,:4]
        gt_labels = target[:,4] + 1   # +1 to account for background class
        
        labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
        
        sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels,
                                                                            positive_count=self.roi_pos_count,
                                                                            total_count=self.roi_loss_count)
        
        sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
        
        # Keep only sampled proposals
        proposals = proposals[sampled_idxs]
        labels = labels[sampled_idxs]
        matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
        regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)  
        
        # ROI pooling and call all layers for prediction
        proposal_roi_pool_feats = roi_align(feat, [proposals], output_size=self.pool_size, spatial_scale=self.spatial_scale, sampling_ratio=2, aligned=True)
        
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = F.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = F.relu(self.fc7(box_fc_6))
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        # cls_scores -> (proposals, num_classes)
        # box_transform_pred -> (proposals, num_classes * 4)
        ##############################################
        
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)

        classification_loss = F.cross_entropy(cls_scores, labels)
        
        # Compute localization loss only for non-background labelled proposals
        fg_proposals_idxs = torch.where(labels > 0)[0]
        # Get class labels for these positive proposals
        fg_cls_labels = labels[fg_proposals_idxs]
        
        localization_loss = F.smooth_l1_loss(
            box_transform_pred[fg_proposals_idxs, fg_cls_labels],
            regression_targets[fg_proposals_idxs],
            beta=1/9,
            reduction="sum",
        )
        localization_loss = localization_loss / labels.numel()

        return {"roi_cls_loss": classification_loss,
                "roi_loc_loss": localization_loss}

