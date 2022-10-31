
import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
import coremltools as ct
# import logging
# logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def main(config, in_model_path, output_model_path, input_shape=(320, 320)):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(in_model_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    model.eval()

    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )

    traced_model = torch.jit.trace(model, dummy_input)
    print(dummy_input.shape)

    # logging.info("convert coreml start.")
    core_model = ct.convert(
        traced_model,
        # inputs=[ct.ImageType(shape=dummy_input.shape, name='input', scale=0.017429, bias=(-103.53 * 0.017429, -116.28 * 0.017507, -123.675 * 0.017125))],
        inputs=[ct.ImageType(shape=dummy_input.shape, name='input', scale=1/(0.226*255.0), bias=[ -0.406/(0.225), -0.485/(0.229), -0.456/(0.224)])],
        outputs=[ct.TensorType(name="output")],
        debug=True
    )
    core_model.save(output_model_path)
    # logging.info("finish convert coreml.")


if __name__ == "__main__":
    cfg_path = 'config/nanodet-plus-m-1.5x_320.yml'
    model_path = 'checkpoint/nanodet-plus-m-1.5x_320.pth'
    mlmodel_path = 'checkpoint/nanodet-plus-m-1.5x_320.mlmodel'
    input_shape = (320, 320)
    load_config(cfg, cfg_path)
    main(cfg, model_path, mlmodel_path, input_shape)
    print("Model saved to:", mlmodel_path)
