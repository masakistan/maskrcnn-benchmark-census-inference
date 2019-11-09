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

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

BUFFER = 15
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


def area(a, b):  # returns None if rectangles don't intersect
    a = Rectangle(*a)
    b = Rectangle(*b)
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def process(coco_demo, img_idx, img_path, out_dir, debug_dir):
    # load image and then run prediction
    try:
        if type(img_path) == list or type(img_path) == tuple:
            image = io.imread(join(*img_path))
        else:
            image = io.imread(img_path)
        if image.shape[0] < 1000 or image.shape[1] < 1000:
            print("ERROR: image has strange dimensions {}".format(image.shape))
            return
    except Exception as e:
        print('ERROR: could not process {} with exception {}'.format(join(*img_path), e))
        return
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
        return

    items = list(zip(scores, labels, boxes))
    items.sort(key = lambda x: x[2][1])
    name_header = list(filter(lambda x: x[1] == 3, items))
    name_col = list(filter(lambda x: x[1] == 4, items))
    items = list(filter(lambda x: x[1] == 2, items))

    #print '\n'.join(map(str,items))

    if type(img_path) == list or type(img_path) == tuple:
        prefix = img_path[-1]
        prefix = prefix[:prefix.rfind('.')]
    else:
        prefix = img_path
        prefix = prefix[prefix.rfind("/") + 1:prefix.rfind(".")]

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

    print("\tINFO: name field avg height is {} with avg x1 {} and x2 {}".format(avg_height, avg_x1, avg_x2))

    # NOTE: corrections
    if len(coords) > 40:
        exp = 50
    elif len(coords) < 5:
        exp = 0
    else:
        exp = 25

    print('\tINFO: initially found', len(coords), 'snippets')
    new_coords = []

    # NOTE: check if the name col header is present, if it is, use that as the first thing
    failed = False
    if len(name_header) > 0:
        coords_check = coords
        pcoord = list(map(int, name_header[0][2].numpy()))
        #print('col header coords:', pcoord)
    else:
        coords_check = coords[1:]
        pcoord = coords[0]
        print("\tOutput status: ERROR! Could not find name column header, giving up")
        failed = True

    if len(name_col) > 0:
        name_col_coord = list(map(int, name_col[0][2].numpy()))
        name_col_orig_end = name_col_coord[3]
        name_col_coord[1] = 0
        name_col_coord[3] = image.shape[0]
    else:
        print("\tOutput status: ERROR! Could not identify the name column, giving up")
        failed = True

    if len(items) <= 0:
        print("\tOutput status: ERROR! found no fields on page, giving up")
        failed = True

    if failed:
        if debug_dir:
            try:
                makedirs(debug_dir)
            except:
                pass
            if exp != len(coords) or len(too_much_overlap) > 0:
                cv2.imwrite(join(debug_dir, prefix + '.jpg'), predictions)
        return None


    fixed_coords = []
    filtered = []
    for idx, ccoord in enumerate(coords_check):

        overlap = area(name_col_coord, ccoord)
        #print('p', pcoord)
        #print('c', ccoord)
        #print('o', overlap)

        if overlap is None:
            filtered.append(ccoord)
            continue

        fixed_coords.append(ccoord)

        if pcoord[3] + BUFFER > ccoord[1]:
            pcoord = ccoord
        else:
            while pcoord[3] + BUFFER <= ccoord[1]:
                y1 = pcoord[3]
                y2 = y1 + avg_height

                pcoord = (avg_x1, y1, avg_x2, y2)
                new_coords.append(pcoord)
            pcoord = ccoord
    coords = fixed_coords + new_coords

    coords.sort(key=lambda x: (x[1] + x[3]) / 2)
    last = coords[-1]
    add_to_end = []
    while last[3] + avg_height - 10 < name_col_orig_end:
        y1 = last[3] - 15
        y2 = y1 + avg_height
        new_coord = avg_x1, y1, avg_x2, y2
        add_to_end.append(new_coord)

        last = new_coord
    coords += add_to_end

    for coord in new_coords:
        x1, y1, x2, y2 = coord
        predictions = cv2.rectangle(
            predictions,
            tuple((x1, y1)),
            tuple((x2, y2)),
            tuple([255, 0, 0]),
            2
        )
        
    for coord in add_to_end:
        x1, y1, x2, y2 = coord
        predictions = cv2.rectangle(
            predictions,
            tuple((x1, y1)),
            tuple((x2, y2)),
            tuple([255, 0, 255]),
            2
            )

    for coord in filtered:
        x1, y1, x2, y2 = coord
        predictions = cv2.rectangle(
            predictions,
            tuple((x1, y1)),
            tuple((x2, y2)),
            tuple([255, 128, 0]),
            2
        )

    print("\tINFO: filtered out {} name fields".format(len(filtered)))

    coords.sort(key=lambda x: (x[1] + x[3]) / 2)
    print('\tINFO: correct amount?', exp == len(coords))
    if len(coords) > exp:
        coords = coords[:exp]
        print("\tINFO: get top {} amount ? {}".format(exp, exp == len(coords)))

    for x1, y1, x2, y2 in coords:
        y1 -= 5
        y2 += 20
        frag = image[y1 : y2, x1 : x2]
        frags.append(frag)


    # NOTE: check to see if any cells overlap too much
    write_debug = False
    too_much_overlap = []
    for idx, coord in enumerate(coords[:-1]):
        c_y2 = coord[3]
        n_y1 = coords[idx + 1][1]
        if c_y2 - n_y1 > avg_height / 2:
            print("\tINFO: found cells that overlap by {}".format(avg_height / 2))
            too_much_overlap.append(coord)

    for coord in too_much_overlap:
        x1, y1, x2, y2 = coord
        predictions = cv2.rectangle(
            predictions,
            tuple((x1, y1)),
            tuple((x2, y2)),
            tuple([128, 0, 255]),
            2
        )

            
    if debug_dir:
        try:
            makedirs(debug_dir)
        except:
            pass
        if exp != len(coords) or len(too_much_overlap) > 0:
            cv2.imwrite(join(debug_dir, prefix + '.jpg'), predictions)


    print('\tINFO: final output of', len(frags), 'snippets')

    print('\tINFO: inserted', len(new_coords), 'cells')
    print("\tINFO: inserted {} cells at the end".format(len(add_to_end)))
    

    #print(out_dir, prefix)
    if out_dir is None:
        return top_predictions, predictions

    img_out_dir = join(out_dir, prefix)
    try:
        makedirs(img_out_dir)
    except:
        print("\tINFO: maybe out dir already exists?")

    for i, frag in enumerate(frags):
        fname = prefix + '_' + str(i) + '.jpg'
        out_path = join(img_out_dir, fname)
        cv2.imwrite(out_path, frag)

    if len(coords) == exp and len(too_much_overlap) == 0:
        print("\tOutput status: PASS")
    else:
        print("\tOutput status: FAIL")

    return top_predictions, predictions
