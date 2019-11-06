'''
usage: <config> <input_dir> <output_dir>
'''

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys, math
import torch
import numpy as np
import cv2
from collections import Counter
from os import makedirs
from os import listdir
from os.path import isfile, join


cats = [
    '__ignore__',
    '_background_',
    'name_col_field',
    'name_col_header',
    'name_col',
    'occupation_col_header',
    'occupation_col_occupation_field',
    'occupation_col_industry_field',
    'occupation_col',
    'veteran_col_header',
    'veteran_col_yes_or_no',
    'veteran_col_war_or_expedition',
    'veteran_col',
    ]

config_file = sys.argv[1]

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    confidence_threshold=0.9,
)
# load image and then run prediction

mypath = sys.argv[2]
img_paths = [(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and (f[-3:] == 'jpg' or f[-3:] == 'png')]

out_dir = sys.argv[3]

for img_idx, img_path in enumerate(img_paths):
    # load image and then run prediction
    print("{}/{} Processing image {}".format(img_idx, len(img_paths), img_path[-1]))
    try:
        image = io.imread(join(*img_path))
        if image.shape[0] < 1000 or image.shape[1] < 1000:
            print("ERROR: image has strange dimensions {}".format(image.shape))
            continue
    except:
        print('ERROR: could not process', join(*img_path))
        continue
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    top_predictions, predictions = coco_demo.run_on_opencv_image(image)
    #plt.imsave(sys.argv[3] + '.jpg', predictions)
    
    counts = Counter()
    for i in top_predictions.get_field('labels').tolist():
        i = cats[i]
        counts[i] += 1

    for cat in cats:
        print('\t', cat, counts[cat])

    boxes = top_predictions.bbox
    labels = top_predictions.get_field("labels")
    scores = top_predictions.get_field("scores")
    data = torch.cat((scores.reshape(-1,1), boxes), 1)

    if len(boxes) == 0:
        print('\tINFO: no cells found')
        continue

    items = list(zip(scores, labels, boxes))
    items = list(filter(lambda x: x[1] == 2, items))
    items.sort(key = lambda x: x[2][1])

    #print '\n'.join(map(str,items))

    prefix = img_path[-1]
    prefix = prefix[:prefix.rfind('.')]
    img_out_dir = join(out_dir, prefix)
    try:
        makedirs(img_out_dir)
    except:
        print("\tINFO: maybe out dir already exists?")

    coords = []
    frags = []
    for idx, (score, label, box) in enumerate(items):
        x1, y1, x2, y2 = map(int, box.numpy())
        coords.append((y1, y2))

        y1 -= 5
        y2 += 20
        frag = image[y1 : y2, x1 : x2]
        frags.append(frag)

    coords = []
    frags = []
    for idx, (score, label, box) in enumerate(items):
        x1, y1, x2, y2 = map(int, box.numpy())
        coords.append((x1, y1, x2, y2))

    avg_height = int(math.ceil(sum([y2 - y1 for x1, y1, x2, y2 in coords]) / len(coords)))
    avg_x1 = int(math.ceil(sum([x1 for x1, y1, x2, y2 in coords]) / len(coords)))
    avg_x2 = int(math.ceil(sum([x2 for x1, y1, x2, y2 in coords]) / len(coords)))

    # NOTE: corrections
    if len(coords) > 40:
        exp = 50
    elif len(coords) < 5:
        exp = 0
    else:
        exp = 25

    print('\tINFO: initially found', len(coords), 'snippets')
    new_coords = []
    pcoord = coords[0]
    for idx, ccoord in enumerate(coords[1:]):

        if pcoord[3] + 5 > ccoord[1]:
            pcoord = ccoord
        else:
            while pcoord[3] + 5 <= ccoord[1]:
                y1 = pcoord[3]
                y2 = y1 + avg_height

                pcoord = (avg_x1, y1, avg_x2, y2)
                new_coords.append(pcoord)
            pcoord = ccoord
    coords += new_coords

    coords.sort(key=lambda x: (x[1] + x[3]) / 2)

    for x1, y1, x2, y2 in coords:
        y1 -= 5
        y2 += 20
        frag = image[y1 : y2, x1 : x2]
        frags.append(frag)

    print('\tINFO: final output of', len(frags), 'snippets')
    print('\tINFO: correct amount?', exp == len(frags))

    print('\tinserted', len(new_coords), 'cells')

    for i, frag in enumerate(frags):
        fname = prefix + '_' + str(i) + '.jpg'
        out_path = join(img_out_dir, fname)
        cv2.imwrite(out_path, frag)
