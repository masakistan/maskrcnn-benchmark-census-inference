from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import cv2, math
from collections import Counter

labels = [
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
image = io.imread(sys.argv[2])
if len(image.shape) == 2:
    image = color.gray2rgb(image)
top_predictions, predictions = coco_demo.run_on_opencv_image(image)
print("total predictions: {}".format(len(top_predictions)))

counts = Counter()
for i in top_predictions.get_field('labels').tolist():
    i = labels[i]
    counts[i] += 1

print('counts')
for label in labels:
    print(label, counts[label])

    
#print(top_predictions)
boxes = top_predictions.bbox
labels = top_predictions.get_field("labels")
scores = top_predictions.get_field("scores")
data = torch.cat((scores.reshape(-1,1), boxes), 1)

items = list(zip(scores, labels, boxes))
items = list(filter(lambda x: x[1] == 2, items))
items.sort(key = lambda x: x[2][1])

# NOTE: corrections
if len(items[2]) > 40:
    exp = 50
elif len(items[2]) < 5:
    exp = 0
else:
    exp = 25

if len(items[2]) != exp:
    diff = len(items[2]) - exp


coords = []
frags = []
for idx, (score, label, box) in enumerate(items):
    x1, y1, x2, y2 = map(int, box.numpy())
    coords.append((x1, y1, x2, y2))

avg_height = int(math.ceil(sum([y2 - y1 for x1, y1, x2, y2 in coords]) / len(coords)))
avg_x1 = int(math.ceil(sum([x1 for x1, y1, x2, y2 in coords]) / len(coords)))
avg_x2 = int(math.ceil(sum([x2 for x1, y1, x2, y2 in coords]) / len(coords)))
print('cell avg height:', avg_height, avg_x1, avg_x2)
    
new_coords = []
pcoord = coords[0]
for idx, ccoord in enumerate(coords[1:]):
    print('prev', pcoord)
    print('curr', ccoord)

    if pcoord[3] + 5 > ccoord[1]:
        pcoord = ccoord
    else:
        while pcoord[3] + 5 <= ccoord[1]:
            print('missing cell?')
            #print(pcoord)
            y1 = pcoord[3]
            y2 = y1 + avg_height

            pcoord = (avg_x1, y1, avg_x2, y2)
            new_coords.append(pcoord)
            print('adding', pcoord)
        pcoord = ccoord
coords += new_coords

coords.sort(key=lambda x: (x[1] + x[3]) / 2)
        
for x1, y1, x2, y2 in coords:
    y1 -= 5
    y2 += 20
    frag = image[y1 : y2, x1 : x2]
    frags.append(frag)

print('inserted', len(new_coords), 'cells')

for i, frag in enumerate(frags):
    cv2.imwrite(sys.argv[3] + str(i) + '.jpg', frag)
#plt.imsave(sys.argv[3] + '.jpg', predictions)
io.imsave(sys.argv[3], predictions)
