import os
import cv2
import math
import torch
import numpy as np

from nanodet.model.module.nms import multiclass_nms
from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv
from nanodet.data.transform.warp import warp_boxes
from nanodet.model.head.gfl_head import Integral, reduce_mean
from nanodet.util.visualization import _COLORS

num_classes = 80
reg_max = 7
distribution_project = Integral(reg_max)
strides = [8, 16, 32, 64]

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]

def post_process(preds, meta):
    """Prediction results post processing. Decode bboxes and rescale
    to original image size.
    Args:
        preds (Tensor): Prediction output.
        meta (dict): Meta info.
    """
    cls_scores, bbox_preds = preds.split(
        [num_classes, 4 * (reg_max + 1)], dim=-1
    )
    result_list = get_bboxes(cls_scores, bbox_preds, meta)

    det_results = {}
    warp_matrixes = (
        meta["warp_matrix"]
        if isinstance(meta["warp_matrix"], list)
        else meta["warp_matrix"]
    )
    img_heights = (
        meta["img_info"]["height"].cpu().numpy()
        if isinstance(meta["img_info"]["height"], torch.Tensor)
        else meta["img_info"]["height"]
    )
    img_widths = (
        meta["img_info"]["width"].cpu().numpy()
        if isinstance(meta["img_info"]["width"], torch.Tensor)
        else meta["img_info"]["width"]
    )
    img_ids = (
        meta["img_info"]["id"].cpu().numpy()
        if isinstance(meta["img_info"]["id"], torch.Tensor)
        else meta["img_info"]["id"]
    )

    # for result, img_width, img_height, img_id, warp_matrix in zip(
    #     result_list, img_widths, img_heights, img_ids, warp_matrixes
    # ):
    for result, warp_matrix in zip(result_list, warp_matrixes):
        img_width, img_height, img_id = 320, 320, 0
        det_result = {}
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_bboxes[:, :4] = warp_boxes(
            det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
        )
        classes = det_labels.detach().cpu().numpy()
        for i in range(num_classes):
            inds = classes == i
            det_result[i] = np.concatenate(
                [
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        det_results[img_id] = det_result
    return det_results


def get_bboxes(cls_preds, reg_preds, img_metas):
    """Decode the outputs to bboxes.
    Args:
        cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
        reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
        img_metas (dict): Dict of image info.

    Returns:
        results_list (list[tuple]): List of detection bboxes and labels.
    """
    device = cls_preds.device
    b = cls_preds.shape[0]
    input_height, input_width = 320, 320
    input_shape = (input_height, input_width)

    featmap_sizes = [
        (math.ceil(input_height / stride), math.ceil(input_width) / stride)
        for stride in strides
    ]
    # get grid cells of one image
    mlvl_center_priors = [
        get_single_level_center_priors(
            b,
            featmap_sizes[i],
            stride,
            dtype=torch.float32,
            device=device,
        )
        for i, stride in enumerate(strides)
    ]
    center_priors = torch.cat(mlvl_center_priors, dim=1)
    dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
    scores = cls_preds.sigmoid()
    result_list = []
    for i in range(b):
        # add a dummy background class at the end of all labels
        # same with mmdetection2.0
        score, bbox = scores[i], bboxes[i]
        padding = score.new_zeros(score.shape[0], 1)
        score = torch.cat([score, padding], dim=1)
        results = multiclass_nms(
            bbox,
            score,
            score_thr=0.05,
            nms_cfg=dict(type="nms", iou_threshold=0.6),
            max_num=100,
        )
        result_list.append(results)
    return result_list


def get_single_level_center_priors( batch_size, featmap_size, stride, dtype, device):
    """Generate centers of a single stage feature map.
    Args:
        batch_size (int): Number of images in one batch.
        featmap_size (tuple[int]): height and width of the feature map
        stride (int): down sample stride of the feature map
        dtype (obj:`torch.dtype`): data type of the tensors
        device (obj:`torch.device`): device of the tensors
    Return:
        priors (Tensor): center priors of a single level feature map.
    """
    h, w = featmap_size
    x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
    y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
    y, x = torch.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = x.new_full((x.shape[0],), stride)
    proiors = torch.stack([x, y, strides, strides], dim=-1)

    return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


def show_result(
    img, dets, class_names, score_thres=0.3, show=True, save_path=None
):
    result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
    if show:
        cv2.imshow("det", result)
    return result


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    all_label = {}
    crop_img = None
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])

    # max_area = 0
    # max_box = 0
    # for box in all_box:
    #     label, x0, y0, x1, y1, score = box
    #     area = (x1-x0) * (y1-y0)
    #     if area > max_area:
    #         max_box = box
    #         max_area = area
    #
    # all_box = [max_box]

    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # if label not in [0,1]: continue
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        crop_img = img[y0:y1 + 1, x0:x1]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )

        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
        # all_label.append(label)
        all_label[label] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1, score)
    return img, all_label, crop_img


def visualize(dets, meta, class_names, score_thres, wait=0):
    result_img, all_label, crop_img = show_result(
        meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
    )
    # print("viz time: {:.3f}s".format(time.time() - time1))
    return result_img, all_label, crop_img

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names