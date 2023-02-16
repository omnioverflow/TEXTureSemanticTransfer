import numpy as np, cv2
from matplotlib import pyplot as plt

# OpenCV Image Denoising
# https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html


id = 6
img = cv2.imread(f'albedo_000{id}.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 14, 42)
cv2.imwrite(f'albedo_out_000{id}.png', cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
plt.subplot(121); plt.imshow(img)
plt.subplot(122); plt.imshow(dst)   
plt.show()
