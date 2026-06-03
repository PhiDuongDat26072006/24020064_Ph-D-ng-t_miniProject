import matplotlib.pyplot as plt
import cupy as cp
from matplotlib import patches
import numpy as np

def Visualization_img_result(pred, gt, img_org):
    num_samples = len(pred)
    for idx in range(num_samples):
        img_matrix = cp.asarray(img_org[idx])
        height_org = img_matrix.shape[0]
        width_org  = img_matrix.shape[1]
        boxes_pred = pred[idx]
        boxes_gt = gt[idx]

        x = cp.sqrt(num_samples)
        x_rounded = cp.round(x)
        if x_rounded < x:
            x = int(x_rounded)
            ax = plt.subplot(x, x + 1, idx + 1)
        else:
            x = int(x_rounded)
            ax = plt.subplot(x, x, idx + 1)

        ax.imshow(img_org[idx])

        for idx, box in enumerate(boxes_pred):
            x_pred = box[2] * width_org
            y_pred = box[3] * height_org
            w_pred = box[4] * width_org
            h_pred = box[5] * height_org

            rect_pred = patches.Rectangle((x_pred - w_pred/2, y_pred - h_pred/2),w_pred, h_pred,
                                          linewidth=2, edgecolor='red', facecolor='none', label='Predicted')
            ax.add_patch(rect_pred)
            ax.axis('off')

        for idx, box in enumerate(boxes_gt):
            x_gt = box[1] * width_org
            y_gt = box[2] * height_org
            w_gt = box[3] * width_org
            h_gt = box[4] * height_org

            rect_gt = patches.Rectangle((x_gt - w_gt/2, y_gt - h_gt/2), w_gt, h_gt, linewidth=2,
                                        edgecolor='green', facecolor='none', label='Ground Truth')
            ax.add_patch(rect_gt)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    return

def Visualization_train_valid_result(training_loss, validating_loss):
    print(len(training_loss))
    print(len(validating_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(cp.array(training_loss).get(), label='Training Loss', color='blue', linewidth=2)
    plt.plot(cp.array(validating_loss).get(), label='Validation Loss', color='red', linewidth=2)

    plt.title('Model Loss Tracking', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()