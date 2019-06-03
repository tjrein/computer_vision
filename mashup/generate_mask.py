from PIL import Image
import numpy as np
import cv2
import imageio
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

height = 200
width = 200

def load_image(filename):
    image = Image.open(filename).resize((height, width))
    return np.float32(image)


#content_img = load_image("./content2.jpg")

test_style_1 = cv2.imread('./bojack_starry.jpg')
test_style_2 = cv2.imread('./bojack_manet.jpg')
test_style_3 = cv2.imread('./bojack_seurat.jpg')

mask_1 = cv2.imread('./bojack_mask1_resize.jpg', 0)
mask_2 = cv2.imread('./bojack_mask2_resize.jpg', 0)

inverse_mask_1 = 255 - mask_1
inverse_mask_2 = 255 - mask_2

test_style_1 = cv2.cvtColor(test_style_1, cv2.COLOR_BGR2RGB)
test_style_2 = cv2.cvtColor(test_style_2, cv2.COLOR_BGR2RGB)
test_style_3 = cv2.cvtColor(test_style_3, cv2.COLOR_BGR2RGB)



#hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
#light_orange = np.array([27, 18, 24])
#dark_orange = np.array([255, 255, ])
#mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result_1 = cv2.bitwise_and(test_style_1, test_style_1, mask=mask_1)
result_2 = cv2.bitwise_and(test_style_2, test_style_2, mask=inverse_mask_1)
result_3 = cv2.bitwise_or(result_1, result_2)

result_4 = cv2.bitwise_and(test_style_3, test_style_3, mask=mask_2)
result_5 = cv2.bitwise_and(result_3, result_3, mask=inverse_mask_2)
result_6 = cv2.bitwise_or(result_4, result_5)
#plt.imshow(mask, cmap='gray')
#plt.imshow(mask, cmap='gray')
#plt.imshow(result)



#plt.subplot(1, 3, 1)
#plt.imshow(result_4)

#plt.subplot(1, 3, 2)
#plt.imshow(result_5)

plt.subplot(1, 1, 1)
plt.imshow(result_6)

plt.show()
