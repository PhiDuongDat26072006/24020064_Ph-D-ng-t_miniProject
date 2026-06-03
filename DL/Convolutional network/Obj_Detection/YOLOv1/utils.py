import cupy as cp
import torch

def Adam(W, B, dW, dB, V_dw, V_db, S_dw, S_db, Beta1, Beta2, num_of_mn_batch,lr):
    num_of_mn_batch += 1

    V_dw = Beta1 * V_dw + (1 - Beta1) * dW
    V_db = Beta1 * V_db + (1 - Beta1) * dB
    S_dw = Beta2 * S_dw + (1 - Beta2) * (dW ** 2)
    S_db = Beta2 * S_db + (1 - Beta2) * (dB ** 2)

    V_dw_corr = V_dw / (1 - Beta1 ** num_of_mn_batch + 1e-8)
    V_db_corr = V_db / (1 - Beta1 ** num_of_mn_batch + 1e-8)
    S_dw_corr = S_dw / (1 - Beta2 ** num_of_mn_batch + 1e-8)
    S_db_corr = S_db / (1 - Beta2 ** num_of_mn_batch + 1e-8)

    W = W - lr * V_dw_corr / (cp.sqrt(S_dw_corr) + 1e-8)
    B = B - lr * V_db_corr / (cp.sqrt(S_db_corr) + 1e-8)

    return W, B, V_dw, V_db, S_dw, S_db

def convert_to_boxes(y_hat):
    S = y_hat.shape[1]
    batch_size = y_hat.shape[0]
    boxes = []
    for batch_idx in range(batch_size):
        boxes_on_img = []
        for i in range(S):
            for j in range(S):
                cell = y_hat[batch_idx,i,j]
                for b in range(2):
                    class_idx = int(cp.argmax(cell[:20]))
                    confidence = cell[20 + b * 5]
                    x = (cell[21 + b * 5] + j) / S
                    y = (cell[22 + b * 5] + i) / S
                    w = cell[23 + b * 5]
                    h = cell[24 + b * 5]
                    boxes_on_img.append([int(class_idx), float(confidence), float(x), float(y), float(w), float(h)])
        boxes.append(boxes_on_img)
    return boxes

def intersection_of_union(predicted_box, target_box):
    xmin_pred = predicted_box[...,0] - predicted_box[...,2]/2
    xmax_pred = predicted_box[...,0] + predicted_box[...,2]/2
    ymin_pred = predicted_box[...,1] - predicted_box[...,3]/2
    ymax_pred = predicted_box[...,1] + predicted_box[...,3]/2
    xmin_gt = target_box[...,0] - target_box[...,2] / 2
    xmax_gt = target_box[...,0] + target_box[...,2] / 2
    ymin_gt = target_box[...,1] - target_box[...,3] / 2
    ymax_gt = target_box[...,1] + target_box[...,3] / 2

    xmin = torch.max(xmin_pred, xmin_gt)
    xmax = torch.min(xmax_pred, xmax_gt)
    ymin = torch.max(ymin_pred, ymin_gt)
    ymax = torch.min(ymax_pred, ymax_gt)

    width_in = torch.clamp(xmax - xmin, min = 0)
    height_in = torch.clamp(ymax - ymin, min = 0)
    intersection = width_in * height_in

    union = predicted_box[...,2] * predicted_box[...,3] + target_box[...,2] * target_box[...,3] - intersection
    iou = intersection / union
    return iou

def non_max_suppression(y_hat, conf_threshold = 0.7, iou_threshold = 0.5):
    bboxes = convert_to_boxes(y_hat) # bboxes = (bz, 2*s*s, [class,conf,x,y,w,h])
    chosen_bboxes = []

    for img_idx, bboxes_in_img in enumerate(bboxes):
        bboxes_in_img = [box for box in bboxes_in_img if box[1]>conf_threshold]
        bboxes_in_img = sorted(bboxes_in_img, key=lambda x: x[1], reverse=True)

        chosen_bboxes_per_img = []
        while bboxes_in_img:
            best_bbox = bboxes_in_img[0]
            chosen_bboxes_per_img.append(best_bbox)
            # mỗi lần giữ lại các box khác class với bbox_chosen hoặc iou < iou_threshold
            bboxes_in_img = [bbox for bbox in bboxes_in_img[1:]
                            if bbox[0] != best_bbox[0] or
                                intersection_of_union(torch.tensor(bbox[2:]), torch.tensor(best_bbox[2:])) < iou_threshold
                            ]
        chosen_bboxes.append(chosen_bboxes_per_img)
    return chosen_bboxes