from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

config_file = sys.argv[1]

# spacing between fragments in pixels
spacing = 10

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

mypath = sys.argv[2]
img_paths = [(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and (f[-3:] == 'jpg' or f[-3:] == 'png')]

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

    boxes = top_predictions.bbox
    labels = top_predictions.get_field("labels")
    scores = top_predictions.get_field("scores")
    data = torch.cat((scores.reshape(-1,1), boxes), 1)
    #print 'boxes:', boxes
    #print data[labels == 3]

    cats = ['background', 'background', 'medcert', 'cod', 'contrib']
    #print 'labels:', labels

    items = list(zip(scores, labels, boxes))
    items = list(filter(lambda x: x[1] == 3, items))
    items.sort(key = lambda x: x[2][1])
    #print '\n'.join(map(str,items))
    frags = []
    for idx, (score, label, box) in enumerate(items):
        #print cats[label], label, score, box.numpy(), type(box)
        x1, y1, x2, y2 = map(int, box.numpy())
        frag = image[y1 : y2, x1 : x2]
        frags.append(frag)
        #plt.imsave(sys.argv[3] + '_' + str(idx) + '.jpg', frag)

    if len(frags) == 0:
        continue
    heights, widths, _ = zip(*(i.shape for i in frags))

    total_width = sum(widths) + (spacing * (len(frags) - 1))
    #print 'total width', total_width
    max_height = max(heights)
    #print widths, heights

    #new_im = Image.new('RGB', (total_width, max_height))
    new_im = np.full((max_height, total_width, 3), 255, np.uint8)

    x_offset = 0
    for idx, im in enumerate(frags):
      #new_im.paste(im, (x_offset,0))
      #print im.shape, x_offset, im.shape[0], im.shape[1]
      start_y = max_height - im.shape[0]
      new_im[start_y:, (idx * spacing) + x_offset : (idx * spacing) + x_offset + im.shape[1], :] = im
      x_offset += im.shape[1]

    cv2.imwrite(join(sys.argv[3], img_path[1]), new_im)
