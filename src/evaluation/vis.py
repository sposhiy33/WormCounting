import os
import cv2
import numpy as np
import torchvision.transforms as standard_transforms

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, pred, vis_dir, eval=False, targets=None, class_labels=None, img_name=None):
    """
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    class_labels -> list: [num_preds] -- predicited class of each predicted point
    """
    if samples.ndim < 4:
        samples = samples.unsqueeze(0)

    try: gts = [targets["point"].tolist()]
    except: print("targets not specified, performing pure eval") 

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose(
        [
            DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            standard_transforms.ToPILImage(),
        ]
    )
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 5

        # Generate and save prediction images
        for i, p in enumerate(pred[idx]):
            if class_labels is not None:
                if class_labels[i] == 0:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (0, 255, 0), -1
                    )
                elif class_labels[i] == 1:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (255, 0, 0), -1
                    )
                elif class_labels[i] == 2:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
                    )
                else:
                    pass
            else:
                sample_pred = cv2.circle(
                    sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
                )

        try: name = targets["image_id"]
        except: name = img_name

        if eval:
            file_name = "{}_{}_pred.jpg".format(int(name), len(pred[idx]))
        else: file_name = "{}_gt_{}_pred_{}_pred.jpg".format(int(name), len(gts[idx]), len(pred[idx])) 
        cv2.imwrite(
            os.path.join(
                vis_dir,
                file_name,
            ),
            sample_pred,
        )

        # if performing pure eval, do not continue
        # saving the groun truth images

        # draw gt
        if not eval:
            for t in gts[idx]:
                sample_gt = cv2.circle(
                    sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1
                )

            cv2.imwrite(
                os.path.join(
                    vis_dir,
                    "{}_gt_{}_pred_{}_gt.jpg".format(
                        int(name), len(gts[idx]), len(pred[idx])
                    ),
                ),
                sample_gt,
            )
