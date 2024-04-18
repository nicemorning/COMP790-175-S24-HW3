from isp_common import *
import sys


x = io.imread('..\\data\\baby.tiff')
x = linearization(x, 0, 16383)
x = camera_preset_wb(x, 1.628906,1.000000,1.386719)
x = demosaicing(x)
x = color_space_correction(x)
x = bright_adjustment(x, 2.5)
y = gamma_enc(x)


y =  (y * 256).astype(np.uint8)
io.imsave('baby_test.png', y)

q_list = [90, 85, 80, 70, 65]

for q in q_list:
    jpeg_fname = f'baby_test_{q}.jpg'
    print(f'Write {jpeg_fname}')
    io.imsave(jpeg_fname, y, quality=q)


z1 = io.imread('baby_test.png')

plt.subplot(3, 2, 1) # draw first image
plt.imshow(z1)
plt.title('baby_test.png')

i =  1    
for q in q_list:
    jpeg_fname = f'baby_test_{q}.jpg'
    print(f'Read {jpeg_fname}')
    z2 = io.imread(jpeg_fname)
    
    i += 1
    plt.subplot(3, 2, i)
    plt.imshow(z2)
    plt.title(f'{jpeg_fname} with Quality:{q}')


plt.show()


