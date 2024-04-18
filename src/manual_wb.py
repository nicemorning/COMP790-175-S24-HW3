from isp_common import *

def manual_white_balance(raw_img, x, y, n):
    # Make a copy, so that the processing will not effect input 
    I =  raw_img.copy()
    # Extract the R, G1, G2, B data from the raw data
    # x1, y1, x2, y2 
    
    x1 = x
    x2 = x + n 
    y1 = y
    y2 = y + n
    
    # The raw_img read from skimage is in (height, width) format
    # corresponding to (y,x)
    p = I[y1:y2,x1:x2]
    
    #Apply camera preset white banlance and demosaicing to display color
    #image of the patch 
    patch = camera_preset_wb(p, 1.628906,1.000000,1.386719)
    patch_rgb = demosaicing(patch) 
 
    plt.figure(figsize=(6, 6))
    plt.subplot(1,2,1)
    #display patch pattern
    plt.imshow(patch, cmap='gray')
    plt.title(f'Patch at {(x, y)}')
    plt.subplot(1,2,2)
    # display patch color
    plt.imshow(patch_rgb)
    plt.show()
    
    
    r = p[0::2, 0::2]
    g1 = p[0::2, 1::2]
    g2 = p[1::2, 0::2]
    b = p[1::2, 1::2] 
    
    # Average color
    mean_r = np.mean(r)
    mean_g = (np.mean(g1) + np.mean(g2)) / 2
    mean_b = np.mean(b)
    
    print(f'Mean color: R({mean_r}) G({mean_g}) B({mean_b})')
    # Assume g_scale is one
    r_scale = mean_g / mean_r
    g_scale = 1
    b_scale = mean_g / mean_b 
    
    print(f'RGB scale:{r_scale, g_scale, b_scale}')
    
    # Apply scale,  Red(0, 0)
    I[0::2, 0::2] *= r_scale
    # Blue (1,1)
    I[1::2, 1::2] *= b_scale
    
    #Normalize I 
    I /= np.max(I)     
    return I

patch_list = [(4000, 50, 20), (1840, 0, 20),(2680, 2200, 20)] 

i = 0 
for (x, y, n) in patch_list:
    i += 1
    z = io.imread('..\\data\\baby.tiff')
    z = linearization(z, 0, 16383)
    z = manual_white_balance(z, x, y, n)
    z = demosaicing(z)
    z = color_space_correction(z)
    z = bright_adjustment(z, 2.5)
    z1 = gamma_enc(z)

    plt.subplot(1, 2, 1) 
    plt.imshow(z1)
    plt.imsave(f'baby_wb_patch{i}.png', z1)
    plt.title('Manual White Balance')

    z2 =  io.imread('..\\data\\baby.jpeg')
    plt.subplot(1, 2, 2) 
    plt.imshow(z2)
    plt.title('Reference')
    plt.show()
