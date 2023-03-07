"""
Extract CLIP features from an image
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
import types
import warnings

import sys
sys.path.insert(1, "/home/evelyn/Desktop/embodied/concept-fusion/Mask2Former")

import cv2
import numpy as np
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from natsort import natsorted
from textwrap import wrap
from tqdm import tqdm, trange

# import clip
# import open_clip

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.colormap import random_color
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


def get_new_pallete(num_colors: int) -> torch.Tensor:
    """Create a color pallete given the number of distinct colors to generate.

    Args:
        num_colors (int): Number of colors to include in the pallete

    Returns:
        torch.Tensor: Generated color pallete of shape (num_colors, 3)
    """
    pallete = []
    # The first color is always black, so generate one additional color
    # (we will drop the black color out)
    for j in range(num_colors + 1):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        if j > 0:
            pallete.append([r, g, b])
    return torch.Tensor(pallete).float() / 255.0


def get_bbox_around_mask(mask):
    # mask: (img_height, img_width)
    # compute bbox around mask
    bbox = None
    nonzero_inds = torch.argwhere(mask)  # (num_nonzero, 2)
    if nonzero_inds.numel() == 0:
        topleft = [0, 0]
        botright = [mask.shape[0], mask.shape[1]]
        bbox = (topleft[0], topleft[1], botright[0], botright[1])  # (x0, y0, x1, y1)
    else:
        topleft = nonzero_inds.min(0)[0]  # (2,)
        botright = nonzero_inds.max(0)[0]  # (2,)
        bbox = (topleft[0].item(), topleft[1].item(), botright[0].item(), botright[1].item())  # (x0, y0, x1, y1)
    # x0, y0, x1, y1
    return bbox, nonzero_inds


def sample_bboxes_around_bbox(bbox, img_height, img_width, scales=[2, 4]):
    # bbox: (x0, y0, x1, y1)
    # scales: scale factor of resultant bboxes to sample
    x0, y0, x1, y1 = bbox
    bbox_height = x1 - x0 + 1
    bbox_width = y1 - y0 + 1
    ret_bboxes = []
    for sf in scales:
        # bbox with Nx size of original must be expanded by size (N-1)x
        # (orig. size = x; new added size = (N-1)x; so new size = (N-1)x + x = Nx)
        # we add (N-1)x // 2  (for each dim; i.e., x = height for x dim; x = width for y dim)
        assert sf >= 1, "scales must have values greater than or equal to 1"
        pad_height = int(math.floor((sf - 1) * bbox_height / 2))
        pad_width = int(math.floor((sf - 1) * bbox_width / 2))
        x0_new, y0_new, x1_new, y1_new = 0, 0, 0, 0
        x0_new = x0 - pad_height
        if x0_new < 0:
            x0_new = 0
        x1_new = x1 + pad_height
        if x1_new >= img_height:
            x1_new = img_height - 1
        y0_new = y0 - pad_width
        if y0_new < 0:
            y0_new = 0
        y1_new = y1 + pad_width
        if y1_new >= img_width:
            y1_new = img_width - 1
        ret_bboxes.append((x0_new, y0_new, x1_new, y1_new))
    return ret_bboxes



# IMGFILE = "otherexamples/caterpillar/007.png"
# IMGFILE = "paperviz/krishnas-tabletop.png"
# IMGFILE = "/home/krishna/data/azurekinect/zerofusion-tabletop/seq03/color/00000.jpg"
IMGFILE = "demo-6.png"
MASK_LOAD_FILE = "tmpmask.pt"
# GLOBAL_FEAT_SAVE_FILE = "demo-6-output.npy"
# SEMIGLOBAL_FEAT_SAVE_FILE = "global_feat_to_all_filtered_masks.pt"
SEMIGLOBAL_FEAT_SAVE_FILE = "tmpsemiglobal.pt"
GLOBAL_FEAT_LOAD_FILE = "demo-6-output.npy"
# OPENSEG_FEAT_LOAD_FILE = "tmpopenseg.pt"
# LOCAL_FEAT_SAVE_FILE_2X = "1/per_mask_feat_2x.pt"
# LOCAL_FEAT_SAVE_FILE_4X = "1/per_mask_feat_4x.pt"
LOAD_IMG_HEIGHT = 512
LOAD_IMG_WIDTH = 512
TGT_IMG_HEIGHT = 512
TGT_IMG_WIDTH = 512
# FEAT_DIM = 1024

# MASK2FORMER_CONFIG_FILE = "/home/krishna/code/Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
# MASK2FORMER_WEIGHTS_FILE = "/home/krishna/code/gradslam-foundation/examples/checkpoints/mask2former/instance/model_final_e5f453.pkl"



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":

    torch.autograd.set_grad_enabled(False)

    # mp.set_start_method("spawn", force=True)
    # cfgargs = types.SimpleNamespace()
    # cfgargs.imgfile = IMGFILE
    # cfgargs.config_file = MASK2FORMER_CONFIG_FILE
    # cfgargs.opts = ["MODEL.WEIGHTS", MASK2FORMER_WEIGHTS_FILE]

    # cfg = setup_cfg(cfgargs)
    # demo = VisualizationDemo(cfg)
    # img = read_image(IMGFILE, format="BGR")
    # predictions, visualized_output = demo.run_on_image(img)
    # masks = torch.nn.functional.interpolate(
    #     predictions["instances"].pred_masks.unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest"
    # )
    # torch.save(masks[0].detach().cpu(), MASK_LOAD_FILE)
    # plt.imshow(visualized_output.get_image()[:, :, ::-1])
    # plt.show()
    # plt.close("all")

    """
    Load masks, sample boxes
    """
    
    img = cv2.imread(IMGFILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = cv2.resize(img, (LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT))
    img = torch.from_numpy(img)

    # OPENCLIP_MODEL = "ViT-H-14"  # "ViT-bigG-14"
    # OPENCLIP_DATA = "laion2b_s32b_b79k"  # "laion2b_s39b_b160k"
    # print("Initializing model...")
    # model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL, OPENCLIP_DATA)
    # model.cuda()
    # model.eval()
    # tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    # """
    # Extract and save global feat vec
    # """
    # global_feat = None
    # with torch.cuda.amp.autocast():
    #     print("Extracting image features...")
    #     _img = preprocess(Image.open(IMGFILE)).unsqueeze(0)
    #     imgfeat = model.encode_image(_img.cuda())
    #     imgfeat /= imgfeat.norm(dim=-1, keepdim=True)
    #     tqdm.write(f"Image feature dims: {imgfeat.shape} \n")
    #     tqdm.write(f"Saving to {GLOBAL_FEAT_SAVE_FILE} \n")
    #     torch.save(imgfeat.detach().cpu().half(), GLOBAL_FEAT_SAVE_FILE)

    # global_feat = torch.load(GLOBAL_FEAT_LOAD_FILE)

    global_feat = torch.tensor(np.load("demo-6-output.npy"))
    global_feat = global_feat.half().cuda()
    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
    FEAT_DIM = global_feat.shape[-1]

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    """
    Extract per-mask features
    """
    # Output feature vector
    outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, FEAT_DIM, dtype=torch.half)

    print(f"Loading instance masks {MASK_LOAD_FILE}...")
    mask = torch.load(MASK_LOAD_FILE).unsqueeze(0)  # 1, num_masks, H, W
    mask = torch.nn.functional.interpolate(mask, [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest")
    num_masks = mask.shape[-3]
    pallete = get_new_pallete(num_masks)

    rois = []
    roi_similarities_with_global_vec = []
    roi_sim_per_unit_area = []
    feat_per_roi = []
    roi_nonzero_inds = []

    for _i in range(num_masks):

        # viz = torch.zeros(IMG_HEIGHT, IMG_WIDTH, 3)
        curmask = mask[0, _i]
        bbox, nonzero_inds = get_bbox_around_mask(curmask)
        x0, y0, x1, y1 = bbox
        # viz[x0:x1, y0:y1, 0] = 1.0

        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        img_area = LOAD_IMG_WIDTH * LOAD_IMG_HEIGHT
        iou = bbox_area / img_area

        if iou < 0.005:
            continue

        # if iou < 0.25:
        #     bboxes = sample_bboxes_around_bbox(bbox, LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, scales=[4])
        #     _x0, _y0, _x1, _y1 = bboxes[0]
        #     img_roi = img[_x0:_x1, _y0:_y1]
        #     img_roi = Image.fromarray(img_roi.detach().cpu().numpy())
        #     img_roi = preprocess(img_roi).unsqueeze(0).cuda()
        #     roifeat = model.encode_image(img_roi)
        #     roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
        #     outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] += roifeat[0].half().detach().cpu()
        #     outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] = torch.nn.functional.normalize(
        #         outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]].float(), dim=-1
        #     ).half()

        # per-mask features
        img_roi = img[x0:x1, y0:y1]
        img_roi = Image.fromarray(img_roi.detach().cpu().numpy())
        img_roi = preprocess(img_roi).unsqueeze(0).cuda()
        roifeat = model.encode_image(img_roi)
        roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
        feat_per_roi.append(roifeat)
        roi_nonzero_inds.append(nonzero_inds)

        _sim = cosine_similarity(global_feat, roifeat)

        rois.append(torch.tensor(list(bbox)))
        roi_similarities_with_global_vec.append(_sim)
        roi_sim_per_unit_area.append(_sim)# / iou)
        # print(f"{_sim.item():.3f}, {iou:.3f}, {_sim.item() / iou:.3f}")


    # """
    # global_clip_plus_mask_weighted
    # # """
    rois = torch.stack(rois)
    scores = torch.cat(roi_sim_per_unit_area).to(rois.device)
    # nms not implemented for Long tensors
    # nms on CUDA is not stable sorted; but the CPU version is
    retained = torchvision.ops.nms(rois.float().cpu(), scores.cpu(), iou_threshold=1.0)
    feat_per_roi = torch.cat(feat_per_roi, dim=0)  # N, 1024
    
    print(f"retained {len(retained)} masks of {rois.shape[0]} total")
    retained_rois = rois[retained]
    retained_scores = scores[retained]
    retained_feat = feat_per_roi[retained]
    retained_nonzero_inds = []
    for _roiidx in range(retained.shape[0]):
        retained_nonzero_inds.append(roi_nonzero_inds[retained[_roiidx].item()])
    
    # viz = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, 3)

    mask_sim_mat = torch.nn.functional.cosine_similarity(
        retained_feat[:, :, None], retained_feat.t()[None, :, :]
    )
    mask_sim_mat.fill_diagonal_(0.)
    mask_sim_mat = mask_sim_mat.mean(1)  # avg sim of each mask with each other mask
    softmax_scores = retained_scores.cuda() - mask_sim_mat
    softmax_scores = torch.nn.functional.softmax(softmax_scores, dim=0)
    print(softmax_scores)
    for _roiidx in range(retained.shape[0]):
        _weighted_feat = softmax_scores[_roiidx] * global_feat + (1 - softmax_scores[_roiidx]) * retained_feat[_roiidx]
        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
        _rand_rgb = random_color(rgb=True, maximum=1)
        # viz[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1], 0] = _rand_rgb[0].item()
        # viz[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1], 1] = _rand_rgb[1].item()
        # viz[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1], 2] = _rand_rgb[2].item()
        outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]] = torch.nn.functional.normalize(
            outfeat[retained_nonzero_inds[_roiidx][:, 0], retained_nonzero_inds[_roiidx][:, 1]].float(), dim=-1
        ).half()

    outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
    outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
    outfeat = torch.nn.functional.interpolate(outfeat, [TGT_IMG_HEIGHT, TGT_IMG_WIDTH], mode="nearest")
    outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
    outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
    outfeat = outfeat[0].half() # --> H, W, feat_dim

    # __img = cv2.imread(IMGFILE)
    # __img = cv2.cvtColor(__img, cv2.COLOR_BGR2RGB)
    # __img = cv2.resize(__img, (LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT))
    # # plt.imshow(0.5 * viz.detach().cpu().numpy())
    # # plt.imshow(__img.astype(float) / 255.0)
    # plt.imshow(0.5 * viz.detach().cpu().numpy() + 0.5 * (__img.astype(float) / 255.0))
    # plt.show()
    # plt.close("all")

    print(SEMIGLOBAL_FEAT_SAVE_FILE)
    torch.save(outfeat, SEMIGLOBAL_FEAT_SAVE_FILE)


    """
    Test aggregation
    """

    openseg_feat = torch.load(OPENSEG_FEAT_LOAD_FILE)
    openseg_feat = openseg_feat.float().cuda()  # (1, 768, H, W)
    openseg_feat = torch.nn.functional.interpolate(openseg_feat, [TGT_IMG_HEIGHT, TGT_IMG_WIDTH], mode="nearest")
    openseg_feat = openseg_feat.permute(0, 2, 3, 1)[0]  # (H, W, 768)

    print("Loading CLIP model for OpenSeg (ViT-L/14@336px)...")
    clip_model_for_openseg, _ = clip.load("ViT-L/14@336px")
    openseg_text_encoder = clip_model_for_openseg.encode_text

    semi_global_feat = torch.load(SEMIGLOBAL_FEAT_SAVE_FILE)  # H, W, 1024
    semi_global_feat = semi_global_feat.float().cuda()
    
    _simfunc_openseg = torch.nn.CosineSimilarity(dim=-1)
    _simfunc_sg = torch.nn.CosineSimilarity(dim=-1)

    while True:
        # Prompt user whether or not to continue
        prompt_text = input("Type a prompt ('q' to quit): ")
        if prompt_text == "q":
            break

        text_for_openseg = clip.tokenize(prompt_text)
        text_for_openseg = text_for_openseg.cuda()
        textfeat_openseg = openseg_text_encoder(text_for_openseg)  # (1, 768)
        textfeat_openseg /= textfeat_openseg.norm(dim=-1, keepdim=True)
        textfeat_openseg = textfeat_openseg.unsqueeze(0)  # --> (1, 1, 1024)

        text = tokenizer([prompt_text])
        textfeat = model.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)  # --> (1, 1024)
        textfeat = textfeat.unsqueeze(0)  # --> (1, 1, 1024)
        
        _sim_openseg = _simfunc_openseg(openseg_feat, textfeat_openseg)
        _sim_semiglobal = _simfunc_sg(semi_global_feat, textfeat)
        
        print(_sim_openseg.max().item(), _sim_openseg.min().item(), _sim_openseg.mean().item())
        print(_sim_semiglobal.max().item(), _sim_semiglobal.min().item(), _sim_semiglobal.mean().item())

        fig, ax = plt.subplots(1, 3)
        img_to_show = img.detach()
        img_to_show = img_to_show.cpu().numpy()
        im0 = ax[0].imshow(img_to_show / 255.0)
        ax[0].axis("off")

        colornorm = matplotlib.colors.Normalize(-0.05, 0.3)
        # sm = matplotlib.cm.ScalarMappable(norm=colornorm)
        
        im1 = ax[1].matshow(_sim_openseg.detach().cpu().numpy(), norm=colornorm)
        ax[1].axis("off")
        divider = make_axes_locatable(ax[1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im1, cax=cax, orientation="vertical")
        
        im2 = ax[2].matshow(_sim_semiglobal.detach().cpu().numpy(), norm=colornorm)
        ax[2].axis("off")
        divider = make_axes_locatable(ax[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im2, cax=cax, orientation="vertical")

        # fig.colorbar(sm, ax=ax.ravel().tolist())

        fig.suptitle(prompt_text)

        plt.show()
