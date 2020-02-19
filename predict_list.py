from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import cv2, math
from collections import Counter
from os.path import join

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

out_dir = sys.argv[3]
prefix = sys.argv[4]
success = 0
failure = 0
with open(sys.argv[2], 'r') as fh:
    for line in fh:
        img_path = line.strip()
        base = img_path[img_path.rfind('/') + 1: img_path.rfind('.')]
        print(img_path)
        print('base', base)
        res = process(coco_demo, 0, img_path, None, None)
        if res is None:
            success += 1
            continue
        status, top_predictions, predictions = res
        if status:
            success += 1
        else:
            failure += 1
            #scores = top_predictions.get_field("scores")
        #scores = np.sort(scores.numpy())
        #print(scores)
        out_path = join(out_dir, prefix + base + '.jpg')
        print(out_path)
        io.imsave(out_path, predictions)
print("Success: {}".format(success))
print("Failure: {}".format(failure))
