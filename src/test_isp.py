from isp_common import *
import sys

if len(sys.argv) < 2:
    mode = '1'
else:
    mode = sys.argv[1]
    
print(f'White Balance Mode:{mode}')

x = io.imread('..\\data\\baby.tiff')
x = linearization(x, 0, 16383)

if mode == '1':
    x = camera_preset_wb(x, 1.628906,1.000000,1.386719)
elif mode == '2':
    x = white_world_wb(x)
elif mode == '3':
    x = gray_world_wb(x)
else:
    print(f'Mode:{mode} not supported!')
    sys.exit(-1)

x = demosaicing(x)
x = color_space_correction(x)
x = bright_adjustment(x, 2.5)
y = gamma_enc(x)

z =  io.imread('..\\data\\baby.jpeg')
plt.subplot(1, 2, 1) 
plt.imshow(y)
plt.imsave(f'baby_wb_{mode}.png', y)

plt.subplot(1, 2, 2) 
plt.imshow(z)
plt.show()

