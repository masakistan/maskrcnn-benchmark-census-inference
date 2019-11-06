from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import cv2

config_file = sys.argv[1]

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)
# load image and then run prediction
image = io.imread(sys.argv[2])
if len(image.shape) == 2:
    image = color.gray2rgb(image)
top_predictions, predictions = coco_demo.run_on_opencv_image(image)
#plt.imsave(sys.argv[3] + '.jpg', predictions)
io.imsave('boxes.jpg', predictions)

boxes = top_predictions.bbox
labels = top_predictions.get_field("labels")
scores = top_predictions.get_field("scores")
data = torch.cat((scores.reshape(-1,1), boxes), 1)
print 'boxes:', boxes
print data[labels == 3]

cats = ['background', 'background', 'medcert', 'cod', 'contrib']
print 'labels:', labels

items = list(zip(scores, labels, boxes))
items.sort(key = lambda x: x[2][1])
items = filter(lambda x: x[1] == 3, items)
print '\n'.join(map(str,items))
frags = []
for idx, (score, label, box) in enumerate(items):
    print cats[label], label, score, box.numpy(), type(box)
    x1, y1, x2, y2 = map(int, box.numpy())
    frag = image[y1 : y2, x1 : x2]
    frags.append(frag)
    #plt.imsave(sys.argv[3] + '_' + str(idx) + '.jpg', frag)

heights, widths, _ = zip(*(i.shape for i in frags))

total_width = sum(widths)
max_height = max(heights)
print widths, heights

#new_im = Image.new('RGB', (total_width, max_height))
new_im = np.full((max_height, total_width, 3), 255, np.uint8)

x_offset = 0
for im in frags:
  #new_im.paste(im, (x_offset,0))
  print im.shape, x_offset, im.shape[0], im.shape[1]
  new_im[:im.shape[0], x_offset : x_offset + im.shape[1], :] = im
  x_offset += im.shape[1]

cv2.imwrite(sys.argv[3], new_im)
