import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def check_bayer_pattern(raw_img, x, y, n):
    # The raw_img read from skimage is in (height, width) format
    # corresponding to (y,x)
    sample_pattern = raw_img[y:y+n, x:x+n]
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_pattern, cmap='gray')
    plt.title(f'Sample Bayer Pattern at {(x, y)}')
    plt.colorbar()
    plt.show()

x = io.imread('..\\data\\baby.tiff')
# Check left upper corner  
check_bayer_pattern(x, 0, 0, 8)
# Check red area in the image 
check_bayer_pattern(x, 2584, 2078, 8)


