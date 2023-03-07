import torch
import types

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import matplotlib
import matplotlib.pyplot as plt

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

IMGFILE = "demo-6.png"
MASK_LOAD_FILE = "tmpmask.pt"
LOAD_IMG_HEIGHT = 256
LOAD_IMG_WIDTH = 256
MASK2FORMER_CONFIG_FILE = "/home/evelyn/Desktop/embodied/concept-fusion/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
MASK2FORMER_WEIGHTS_FILE = "model_final_e5f453.pkl"

if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    cfgargs = types.SimpleNamespace()
    cfgargs.imgfile = IMGFILE
    cfgargs.config_file = MASK2FORMER_CONFIG_FILE
    cfgargs.opts = ["MODEL.WEIGHTS", MASK2FORMER_WEIGHTS_FILE]

    cfg = setup_cfg(cfgargs)
    demo = VisualizationDemo(cfg)
    img = read_image(IMGFILE, format="BGR")

    predictions, visualized_output = demo.run_on_image(img)
    masks = torch.nn.functional.interpolate(
        predictions["instances"].pred_masks.unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest"
    )
    torch.save(masks[0].detach().cpu(), MASK_LOAD_FILE)   

    plt.imshow(visualized_output.get_image()[:, :, ::-1])
    plt.show()
    plt.close("all")