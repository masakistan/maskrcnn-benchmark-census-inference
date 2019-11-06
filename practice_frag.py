from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from skimage import io, color
import matplotlib.pyplot as plt
import sys
import numpy as np

config_file = sys.argv[1]

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=200,
    confidence_threshold=0.1,
)
# load image and then run prediction
image = io.imread(sys.argv[2])
if len(image.shape) == 2:
    image = color.gray2rgb(image)

h, w = image.shape[:2]

print h, w
new_h = int(sys.argv[3])
new_w = int(sys.argv[3])

#top = np.random.randint(0, h - new_h)
#left = np.random.randint(0, w - new_w)

frag = 0
for i in range(0, h - new_h, new_h / 3):
    for j in range(0, w - new_w, new_w / 3):
        top = i
        left = j
        print frag, top, left
        image_frag = image[top: top + new_h, left: left + new_w]
        predictions = coco_demo.run_on_opencv_image(image_frag)
        plt.imsave(sys.argv[4] + '.' + str(frag) + '.jpg', predictions)
        frag += 1
