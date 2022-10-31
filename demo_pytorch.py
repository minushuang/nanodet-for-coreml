import os
import time
import cv2
import torch
import glob

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir
from post_utils import post_process, visualize


def main():
    cfg_path = 'config/nanodet-plus-m-1.5x_320.yml'
    model_path = 'checkpoint/nanodet-plus-m-1.5x_320.pth'
    load_config(cfg, cfg_path)

    current_time = time.localtime()
    local_rank = 0
    logger = Logger(local_rank, use_tensorboard=False)

    model = build_model(cfg.model)
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)
    model = model.to('cpu').eval()

    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    img_paths = glob.glob("test_images/*")
    for img_path in img_paths:
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (320,320))

        img_info = {"id": 0}
        img_info["height"] = 320
        img_info["width"] = 320
        img_info["file_name"] = os.path.basename(img_path)
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = pipeline(None, meta, cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        #---- model inference and post process ----
        with torch.no_grad():
            preds = model(meta["img"])
            results = post_process(preds, meta)
            print(results)

        result_image, all_label, crop_img = visualize(results[0], meta, cfg.class_names, 0.35)

        print(result_image.shape)
        print( all_label, '\n')
        save_folder = os.path.join(
            'workspace/nanodet_m', time.strftime("%Y_%m_%d_%H_%M_%S_pytorch", current_time)
        )
        print(save_folder)
        mkdir(local_rank, save_folder)
        # os.mkdir(save_folder)
        save_file_name = os.path.join(save_folder, os.path.basename(img_path))
        print(save_file_name)
        cv2.imwrite(save_file_name, result_image)




if __name__ == "__main__":
    main()
