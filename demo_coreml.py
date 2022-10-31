import glob
import os
import time

import cv2
import torch
import numpy as np
from PIL import Image
import coremltools as ct

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.util import cfg, load_config
from nanodet.util.path import mkdir
from post_utils import post_process, visualize


def main():
    cfg_path = 'config/nanodet-plus-m-1.5x_320.yml'
    mlmodel_path = 'checkpoint/nanodet-plus-m-1.5x_320.mlmodel'
    load_config(cfg, cfg_path)

    current_time = time.localtime()
    local_rank = 0

    model = ct.models.MLModel(mlmodel_path)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    img_paths = glob.glob("test_images/*")
    for img_path in img_paths:
        print(img_path)
        example_image = Image.open(img_path).resize((320, 320)).convert('RGB')

        img_info = {"id": 0}
        img_info["height"] = 320
        img_info["width"] = 320
        img_info["file_name"] = os.path.basename(img_path)
        meta = dict(img_info=img_info, raw_img=np.array(example_image), img=np.array(example_image))
        meta = pipeline(None, meta, cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        # ---------model inference --------
        preds = model.predict({'input': example_image})
        preds = torch.from_numpy(preds['output'])
        # --------- preds post process --------
        results = post_process(preds, meta)
        # --------- visualize --------
        result_image, all_label, crop_img = visualize(results[0], meta, cfg.class_names, 0.35)

        print(result_image.shape)
        print( all_label, '\n')
        save_folder = os.path.join('workspace/nanodet_m', time.strftime("%Y_%m_%d_%H_%M_%S_coreml", current_time))
        mkdir(local_rank, save_folder)
        save_file_name = os.path.join(save_folder, os.path.basename(img_path))
        cv2.imwrite(save_file_name, result_image[:,:,::-1])

if __name__ == "__main__":
    main()
