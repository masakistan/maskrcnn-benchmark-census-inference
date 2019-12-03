from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import cv2, math
from collections import Counter

from census_name import process, cats

config_file = sys.argv[1]

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.7,
)

if len(sys.argv) > 4:
    out_dir = sys.argv[4]
else:
    out_dir = None

if len(sys.argv) > 5:
    debug_dir = sys.argv[5]
else:
    debug_dir = None

top_predictions, predictions = process(coco_demo, 0, sys.argv[2], out_dir, debug_dir)
#scores = top_predictions.get_field("scores")
#scores = np.sort(scores.numpy())
#print(scores)
io.imsave(sys.argv[3], predictions)
