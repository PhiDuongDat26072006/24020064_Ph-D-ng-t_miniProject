from utils import *
class Loss:
    def __init__(self, lambda_coord = 5, lambda_noobj = 0.5):
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.pred_boxes = None
        self.gt_boxes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, pred_boxes, gt_boxes):
        self.pred_boxes = torch.Tensor(pred_boxes).to(self.device)
        self.gt_boxes = torch.Tensor(gt_boxes).to(self.device)

        pred_boxes1 = self.pred_boxes[..., 21:25]
        pred_boxes2 = self.pred_boxes[..., 26:30]

        iou_box1 = intersection_of_union(pred_boxes1, self.gt_boxes[..., 21:25])
        iou_box2 = intersection_of_union(pred_boxes2, self.gt_boxes[..., 21:25])
        iou = torch.cat((iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)), dim=0)

        max_iou, best_boxes = torch.max(iou, dim=0)
        exist_boxes = self.gt_boxes[..., 20:21]
        best_boxes = best_boxes.unsqueeze(3)
        #    Regression loss
        tar_boxes = exist_boxes * self.gt_boxes[..., 21:25]
        retained_boxes = exist_boxes * ((1 - best_boxes) * pred_boxes1 + best_boxes * pred_boxes2)

        tar_xy = tar_boxes[..., :2]
        tar_wh = tar_boxes[..., 2:4]
        retained_xy = retained_boxes[..., :2]
        retained_wh = retained_boxes[..., 2:4]
        regression_loss = (
                torch.mean((torch.flatten(retained_xy) - torch.flatten(tar_xy)) ** 2) +
                torch.mean((torch.flatten(torch.sign(retained_wh) * torch.sqrt(torch.abs(retained_wh) + 1e-6))
                           - torch.flatten(torch.sqrt(tar_wh + 1e-6))) ** 2))

        #   Confidence loss
        # obj loss
        tar_conf = self.gt_boxes[..., 20:21]
        retained_conf1 = (1 - best_boxes) * self.pred_boxes[..., 20:21]
        retained_conf2 = best_boxes * self.pred_boxes[..., 25:26]
        obj_loss = torch.mean((torch.flatten(exist_boxes * (retained_conf1 + retained_conf2)) -
                              torch.flatten(exist_boxes * tar_conf)) ** 2)
        # no obj loss
        no_obj_loss = torch.mean((torch.flatten((1 - exist_boxes) * retained_conf1) -
                                 torch.flatten((1 - exist_boxes) * tar_conf)) ** 2)
        no_obj_loss += torch.mean((torch.flatten((1 - exist_boxes) * retained_conf2) -
                                  torch.flatten((1 - exist_boxes) * tar_conf)) ** 2)

        #   Classification loss
        tar_classes = exist_boxes * self.gt_boxes[..., :20]
        retained_classes = exist_boxes * self.pred_boxes[..., :20]
        classification_loss = torch.mean((torch.flatten(retained_classes) -
                                         torch.flatten(tar_classes)) ** 2)

        # Overall loss
        overall_loss = self.lambda_coord * regression_loss + obj_loss + no_obj_loss * self.lambda_noobj + classification_loss

        return overall_loss

    def backward(self):
        n = self.pred_boxes.shape[0] * self.pred_boxes.shape[1] * self.pred_boxes.shape[2]
        # print(f'\nw_pred:{self.pred_boxes[...,23].max(),self.pred_boxes[...,28].max()}, h_pred:{self.pred_boxes[...,24].max(),self.pred_boxes[...,29].max()}')
        # print(f'w_gt:{self.gt_boxes[...,23].max()}, h_pred:{self.pred_boxes[...,24].max()}')
        pred_boxes1 = self.pred_boxes[..., 21:25]
        pred_boxes2 = self.pred_boxes[..., 26:30]
        dZ = torch.zeros(self.pred_boxes.shape).to(self.device)

        iou_box1 = intersection_of_union(pred_boxes1, self.gt_boxes[..., 21:25])
        iou_box2 = intersection_of_union(pred_boxes2, self.gt_boxes[..., 21:25])
        ious = torch.cat((iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)), dim=0)

        max_iou, best_boxes = torch.max(ious, dim=0)
        exist_boxes = self.gt_boxes[..., 20:21]
        best_boxes = best_boxes.unsqueeze(3)

        # dZ Regression
        xy1 = self.pred_boxes[..., 21:23]
        xy2 = self.pred_boxes[..., 26:28]
        wh1 = self.pred_boxes[..., 23:25]
        wh2 = self.pred_boxes[..., 28:30]
        xy_tar = exist_boxes * self.gt_boxes[..., 21:23]
        wh_tar = exist_boxes * self.gt_boxes[..., 23:25]

        dZ[..., 23:25] = exist_boxes * (1 - best_boxes) * (self.lambda_coord *
                          ((torch.sign(wh1) * torch.sqrt(torch.abs(wh1) + 1e-6)) - torch.sqrt(wh_tar + 1e-6)) /
                          torch.sqrt(torch.abs(wh1) + 1e-6)) / n

        dZ[..., 28:30] = exist_boxes * best_boxes * (self.lambda_coord *
                          ((torch.sign(wh2) * torch.sqrt(torch.abs(wh2) + 1e-6)) - torch.sqrt(wh_tar + 1e-6)) /
                          torch.sqrt(torch.abs(wh2) + 1e-6)) / n

        dZ[..., 21:23] = self.lambda_coord * exist_boxes * (1 - best_boxes) * 2 * (xy1 - xy_tar) / n
        dZ[..., 26:28] = self.lambda_coord * exist_boxes * best_boxes * 2 * (xy2 - xy_tar) / n

        # dZ Confidence
            # dZ obj
        dZ[..., 20:21] = 2 / n * exist_boxes * (1 - best_boxes) * (self.pred_boxes[..., 20:21] -
                                                                self.gt_boxes[..., 20:21])

        dZ[..., 25:26] = 2 / n * exist_boxes * best_boxes * (self.pred_boxes[..., 25:26] -
                                                          self.gt_boxes[..., 20:21])
            # dZ no obj
        dZ[..., 20:21] += 2 / n * (1 - exist_boxes) * (1 - best_boxes) * (self.pred_boxes[..., 20:21] -
                                                                      self.gt_boxes[..., 20:21]) * self.lambda_noobj

        dZ[..., 25:26] += 2 / n * (1 - exist_boxes) * best_boxes * (self.pred_boxes[..., 25:26] -
                                                                self.gt_boxes[..., 20:21]) * self.lambda_noobj

        # dZ Classification
        dZ[...,:20] = 2 / n * exist_boxes * (self.pred_boxes[..., :20] - self.gt_boxes[..., :20])

        return dZ