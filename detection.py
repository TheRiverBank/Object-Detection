import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from model import Model
from dataProcessing import resize_img

img = cv2.imread('./images/horse_detection_test.jpg')
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rect_max_x, rect_max_y, _ = np.shape(img)
rect_min_x, rect_min_y = 35, 35

scaling_x = rect_max_x//10
scaling_y = rect_max_y//10

def get_rects(areas, max_rects):
    scale_x = areas[0][1] // 10
    scale_y = areas[0][3] // 10

    rects = list()

    for area in areas:
        start_x, end_x, start_y, end_y = area

        for i in range(0, max_rects):
            l_x = (i * scale_x)+start_x
            l_y = (i * scale_y)+start_y
            h_x = end_x - (i * scale_x)
            h_y = end_y - (i * scale_y)

            if l_x >= h_x or l_y >= h_y:
                print("Cannot crop any further. Exiting")
                break

            rects.append((h_x, l_x, h_y, l_y))

    return rects

# Search 1
def single_split(img, max_rects: int):
    rects = ((0, img.shape[0], 0, img.shape[1]), )
    r = get_rects(rects, max_rects)
    return r

def quad_split(img, max_rects: int):
    # Split image into four regions
    m, n, _ = np.shape(img)

    img_center_x = m // 2
    img_center_y = n // 2

    _rect_upper_left = (0, img_center_x, 0, img_center_y)
    _rect_lower_right = (img_center_x, m, img_center_y, n)
    _rect_upper_right = (0, img_center_x, img_center_y, n)
    _rect_lower_left = (img_center_x, m, 0, img_center_y)

    area = (_rect_upper_left, _rect_upper_right, _rect_lower_left, _rect_lower_right, (0, m, 0, n))
    rects = get_rects(area, max_rects)

    #plot_rects(img, rects)

    return rects

def plot_rects(img, rects):
    #for area in rects:
    for start_y, end_y, start_x, end_x in rects:
        cv2.rectangle(img_clone, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    img_larger = cv2.resize(img, (int(img.shape[1])*2, int(img.shape[0]*2)))
    cv2.imshow("rects", img_larger)
    cv2.waitKey(0)


img_clone = img.copy()
img_clone = cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB)

single_split_rects = single_split(img, 10)
quad_slpit_rects = quad_split(img_clone, 10)
print(quad_slpit_rects)

m = Model()
cnn = m.get_model()
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
for i in quad_slpit_rects:
    img_ = img[i[1]:i[0], i[3]:i[2]]
    img_ = resize_img(img_)
    pred = cnn.predict(img_)
    print(pred)

