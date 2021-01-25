import numpy as np # 科学运算库

import cv2 # 计算机视觉库

image_list = [
    [[0, 0, 255], [0, 0, 255]],
    [[0, 255, 0], [0, 255, 0]],
    [[255, 0, 0], [255, 0, 0]]
]

image_array = np.array(image_list)

cv2.imwrite("img/demo3x2.png",image_array)