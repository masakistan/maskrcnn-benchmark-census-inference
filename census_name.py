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
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]

#from sklearn.linear_model import LinearRegression


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


def calc_area(a):
    a = Rectangle(*a)
    x = a.xmax - a.xmin
    y = a.ymax - a.ymin
    return x * y

def calc_overlap(a, b):  # returns None if rectangles don't intersect
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
            img_path_str = join(*img_path)
        else:
            image = io.imread(img_path)
            img_path_str = img_path
        if image.shape[0] < 1000 or image.shape[1] < 1000:
            print("{} Output status: FAIL! image has strange dimensions {}".format(img_path_str, image.shape))
            return
    except Exception as e:
        print("{} Output status: FAIL! could not process image with exception {}".format(img_path_str, e))
        return
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    top_predictions, predictions = coco_demo.run_on_opencv_image(image)
    #plt.imsave(sys.argv[3] + '.jpg', predictions)

    img_height, img_width = image.shape[:2]
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
        print("{} Output status: WARNING! No cells found, giving up".format(img_path_str))
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
    scores = []

    for idx, (score, label, box) in enumerate(items):
        x1, y1, x2, y2 = map(int, box.numpy())
        scores.append(score)
        coords.append((x1, y1, x2, y2))
    print("\tINFO: average score {}".format(np.mean(scores)))
    if np.mean(scores) < 0.7:
        print("{} Output status: WARNING! Cells have avg score of {}, giving up".format(img_path_str, np.mean(scores)))
        print(debug_dir, prefix)
        if debug_dir:
            cv2.imwrite(join(debug_dir, prefix + '.low_avg_score.jpg'), predictions, encode_param)
        return

    if len(coords) == 0:
        print("{} Output status: WARNING! No cells found, giving up".format(img_path_str))
        return None
    avg_height = int(math.ceil(sum([y2 - y1 for x1, y1, x2, y2 in coords]) / len(coords)))
    avg_x1 = int(math.ceil(sum([x1 for x1, y1, x2, y2 in coords]) / len(coords)))
    avg_x2 = int(math.ceil(sum([x2 for x1, y1, x2, y2 in coords]) / len(coords)))

    print("\tINFO: name field avg height is {} with avg x1 {} and x2 {}".format(avg_height, avg_x1, avg_x2))

    # NOTE: corrections
    if len(coords) > 35:
        exp = 50
    elif len(coords) < 5:
        exp = 0
    else:
        exp = 25

    # NOTE: correct based on image size
    if img_width > 5000 and img_height > 3500:

        exp = 50

    print('\tINFO: initially found', len(coords), 'snippets')
    new_coords = []

    # NOTE: check if the name col header is present, if it is, use that as the first thing
    if len(name_header) > 0:
        coords_check = coords
        header_coord = list(map(int, name_header[0][2].numpy()))
        #print('col header coords:', pcoord)
    else:
        coords_check = coords[1:]
        header_coord = None
        print("\tWARNING: could not find column header, results could be bad!")
        #print("{} Output status: ERROR! Could not find name column header, giving up".format(img_path_str))

    if len(name_col) > 0:
        name_col_coord = list(map(int, name_col[0][2].numpy()))
        name_col_orig_end = name_col_coord[3]
        name_col_coord[1] = 0
        name_col_coord[3] = image.shape[0]
        fixed_coords = []
        #filtered_coords = []
        n_filtered = 0
        #print('name col coord', name_col_coord)
        for coord in coords:
            filtered = False
            overlap = calc_overlap(name_col_coord, coord)
            #print('overlap', overlap, coord)
            if overlap is None:
                #filtered_coords.append(coord)
                filtered = True
            else:
                overlap /= calc_area(coord)
                #print('filtering overlap perc', overlap)
                if overlap < 0.50:
                    #print("filtering")
                    filtered = True
                else:
                    fixed_coords.append(coord)

            if filtered:
                x1, y1, x2, y2 = coord
                predictions = cv2.rectangle(
                    predictions,
                    tuple((x1, y1)),
                    tuple((x2, y2)),
                    tuple([255, 0, 255]),
                    2
                )
                n_filtered += 1

        print("\tFiltered", n_filtered, "name cells")
        coords = fixed_coords
    else:
        name_col_coord = None
        print("\tWARNING! Could not identify the name column, results may be bad")
        #return None

    areas = [calc_area(x) for x in coords]
    areas_med = np.median(areas)
    areas_std = np.std(areas)
    areas_buff = areas_std * 4
    #print(areas)
    #print('areas', areas_med, areas_std)

    if areas_std > 2500:
        n_filtered = 0
        fixed_coords = []
        for area, coord in zip(areas, coords):
            filtered = False

            # NOTE: should we test if it's too big too?
            if area < areas_med - areas_buff:
                filtered = True
            else:
                fixed_coords.append(coord)

            if filtered:
                x1, y1, x2, y2 = coord
                predictions = cv2.rectangle(
                    predictions,
                    tuple((x1, y1)),
                    tuple((x2, y2)),
                    tuple([255, 0, 255]),
                    2
                )
                n_filtered += 1
        print("\tFiltered", n_filtered, "small name cells")
        coords = fixed_coords

    tl_dists_x = []
    tl_dists_y = []
    br_dists_x = []
    br_dists_y = []
    dists = []
    for i, x in enumerate(coords[:-1]):
        if coords[i + 1][1] - x[3] > 10:
            continue
        dists.append(x[3] - coords[i + 1][1])
        tl_dists_x.append(x[0] - coords[i + 1][0])
        tl_dists_y.append(x[1] - coords[i + 1][1])
        br_dists_x.append(x[2] - coords[i + 1][2])
        br_dists_y.append(x[3] - coords[i + 1][3])
        #print(x[3] - coords[i + 1][1])

    dist_med = np.median(dists)
    dist_std = np.std(dists)
    dist_buff = dist_std

    tl_x_med = np.median(tl_dists_x)
    tl_y_med = np.median(tl_dists_y)
    br_x_med = np.median(br_dists_x)
    br_y_med = np.median(br_dists_y)
    #print(tl_dists_x)
    #print(tl_dists_y)
    #print(br_dists_x)
    #print(br_dists_y)
    #print(tl_x_med, tl_y_med, br_x_med, br_y_med)

    #print('\tmedian', dist_med, dist_std)


    # NOTE: try to add in missing fields
    #top_left = [x[:2] for x in coords]
    #bottom_right = [x[2:] for x in coords]
    #br_ys = [[x[1]] for x in bottom_right]
    #br_xs = [x[0] for x in bottom_right]

    #br_reg = LinearRegression().fit(br_ys, br_xs)
    #print(br_reg.score(br_ys, br_xs))
    #start = (br_reg.predict([[0]]), 0)
    #end = (br_reg.predict([[img_height]]), img_height)
    #cv2.line(predictions, start, end, (0, 255, 0), 4)

    #tl_ys = [[x[1]] for x in top_left]
    #tl_xs = [x[0] for x in top_left]
    #tl_reg = LinearRegression().fit(tl_ys, tl_xs)
    #start = (tl_reg.predict([[0]]), 0)
    #end = (tl_reg.predict([[img_height]]), img_height)
    #cv2.line(predictions, start, end, (0, 255, 0), 4)
     #print(items)
    #print(bottom_right)


    #filtered = []
    if dist_std > 0.:
        for i, ccoord in enumerate(coords[:-1]):
            ncoord = coords[i + 1]
            dist = ncoord[1] - ccoord[3]

            #print('dist', dist)
            # NOTE: i guess we only care of the gap is too large
            #if dist > dist_med - dist_buff and dist < dist_med + dist_buff:
            #print(i, 'dist', dist, dist_med, dist_buff)
            if dist < dist_med - dist_buff:
                pass
                #print('moving on')
            else:
                inserted = 0
                #print('inserting')
                # NOTE: i guess we only care of the gap is too large
                #print('starting dist', dist, dist_med, dist_std, dist_med - dist_std)
                while dist > dist_med - dist_buff:
                    #print('\tinsertion', dist, ccoord, ncoord)

                    x1 = ccoord[0] - tl_x_med
                    y1 = ccoord[1] - tl_y_med
                    x2 = ccoord[2] - br_x_med
                    y2 = ccoord[3] - br_y_med

                    ccoord = tuple(map(int, (x1, y1, x2, y2)))

                    overlap = calc_overlap(ccoord, ncoord)
                    if overlap:
                        print('overlap', ccoord, overlap, overlap / calc_area(ncoord))
                        overlap /= calc_area(ncoord)
                        if overlap > 0.5:
                            print('\tbreak')
                            break

                    dist = ncoord[1] - ccoord[3]
                    #print('\tnew', dist, ccoord, ncoord)
                    new_coords.append(ccoord)
                    if inserted > 20:
                        #print("ERROR! Tried inserting too many cells")
                        print("{} Output status: FAIL! added too many cells".format(img_path_str))
                        cv2.imwrite(join(debug_dir, prefix + '.jpg'), predictions, encode_param)
                        return None
                    inserted += 1
        coords = coords + new_coords
        print("\tINFO: size after insert {}".format(len(coords)))

        for coord in new_coords:
            x1, y1, x2, y2 = coord
            #print(x1, y1, x2, y2)
            predictions = cv2.rectangle(
                predictions,
                tuple((x1, y1)),
                tuple((x2, y2)),
                tuple([255, 255, 0]),
                2
            )
    coords.sort(key=lambda x: (x[1] + x[3]) / 2)

    '''
    if debug_dir:
        try:
            makedirs(debug_dir)
        except:
            pass
        if exp != len(coords):
            cv2.imwrite(join(debug_dir, prefix + '.jpg'), predictions, encode_param)
    '''

    #print("\tINFO: filtered out {} name fields".format(len(filtered)))

    # NOTE: try to add missing coords at bottom and top
    if header_coord:
        dist = coords[0][1] - header_coord[3]
        between_header = []
        #print(dist, dist_med, dist_buff)
        #if dist > dist_med - dist_buff and dist < dist_med + dist_buff:
        if dist < 25:
            # NOTE: no need to add cells between header and fields
            add_to_beginning = False
            add_to_end = True
        else:
            add_to_beginning = True
            add_to_end = False

        if add_to_beginning:
            begin_cells = []
            fcoord = [
                    header_coord[0],
                    header_coord[3] - avg_height,
                    header_coord[2],
                    header_coord[3]
                    ]

            # NOTE: create the new cell
            x1 = int(fcoord[0] - tl_x_med)
            y1 = int(fcoord[1] - tl_y_med)
            x2 = int(fcoord[2] - br_x_med)
            y2 = int(fcoord[3] - br_y_med)
            ccoord = tuple(map(int, (x1, y1, x2, y2)))
            overlap = calc_overlap(coords[0], ccoord)

            while overlap is None:
                begin_cells.append(ccoord)
                predictions = cv2.rectangle(
                    predictions,
                    tuple((x1, y1)),
                    tuple((x2, y2)),
                    tuple([0, 128, 128]),
                    2
                    )

                x1 = int(ccoord[0] - tl_x_med)
                y1 = int(ccoord[1] - tl_y_med)
                x2 = int(ccoord[2] - br_x_med)
                y2 = int(ccoord[3] - br_y_med)
                ccoord = tuple(map(int, (x1, y1, x2, y2)))

                overlap = calc_overlap(coords[0], ccoord)

            add_to_end = True
            print("\tINFO: inserted {} cells at the beginning".format(len(begin_cells)))
            coords = begin_cells + coords

        if add_to_end:
            end_cells = []
            while len(coords) + len(end_cells) < exp:
                # NOTE: insert at the end of the column
                if len(end_cells) > 0:
                    lcoord = end_cells[-1]
                else:
                    lcoord = coords[-1]
                x1 = lcoord[0] - tl_x_med
                y1 = lcoord[1] - tl_y_med
                x2 = lcoord[2] - br_x_med
                y2 = lcoord[3] - br_y_med

                if y2 > img_height:
                    print("{} Output status: {}, added cell past boundary of image ({} {})".format(img_path_str, "FAIL", y2, img_height))
                    return False, top_predictions, predictions

                end_cells.append(tuple(map(int, (x1, y1, x2, y2))))
                #print('\tINFO: adding to end', end_cells[-1], len(coords) + len(end_cells))

            for coord in end_cells:
                x1, y1, x2, y2 = coord
                predictions = cv2.rectangle(
                    predictions,
                    tuple((x1, y1)),
                    tuple((x2, y2)),
                    tuple([0, 255, 255]),
                    2
                    )
            print("\tINFO: inserted {} cells at the end".format(len(end_cells)))
            coords += end_cells

    coords.sort(key=lambda x: (x[1] + x[3]) / 2)
    print('\tINFO: correct amount?', exp == len(coords), "({}/{})".format(len(coords), exp))

    for x1, y1, x2, y2 in coords:
        y1 -= 5
        y2 += 20
        frag = image[y1 : y2, x1 : x2]
        frags.append(frag)

    '''

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

    '''

    if debug_dir:
        try:
            makedirs(debug_dir)
        except:
            pass
        if exp != len(coords):
            cv2.imwrite(join(debug_dir, prefix + '.jpg'), predictions, encode_param)


    print('\tINFO: final output of', len(frags), 'snippets')

    status = exp == len(coords)
    print('\tINFO: inserted', len(new_coords), 'cells')

    #print(out_dir, prefix)
    if out_dir is None:
        if status:
            status = "PASS"
        else:
            status = "FAIL"
        print("{} Output status: {}, fragments not written to disk".format(img_path_str, status))
        return status, top_predictions, predictions

    img_out_dir = join(out_dir, prefix)
    try:
        makedirs(img_out_dir)
    except:
        print("\tINFO: maybe out dir already exists?")

    for i, frag in enumerate(frags):
        fname = prefix + '_' + str(i) + '.jpg'
        out_path = join(img_out_dir, fname)
        cv2.imwrite(out_path, frag, encode_param)

    if len(coords) == exp:
        print("{} Output status: PASS".format(img_path_str))
    else:
        print("{} Output status: FAIL".format(img_path_str))

    return status, top_predictions, predictions
