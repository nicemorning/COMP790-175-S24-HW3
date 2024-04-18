import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy.interpolate import interp2d
from skimage.color import rgb2gray


def linearization(raw_img, black_val, white_val):
    # The input raw_img is a uint16 2-D numpy array 
    # clip the raw image using the black and white values
    x = np.clip(raw_img, black_val, white_val)
    # Shift the clipped values so that black_val becomes 0
    x = x - black_val
    # Scale to [0 ,1], data type automatically convert to float64. 
    x = x / (white_val - black_val)
    return x
    

def white_world_wb(raw_img):
    I =  raw_img.copy()
    # Extract the channels
    r = I[0::2, 0::2]
    g1 = I[0::2, 1::2]
    g2 = I[1::2, 0::2]
    b = I[1::2, 1::2] 
    # The brightest value on each channel assumed to be white
    max_r = np.max(r)
    max_g = max(np.max(g1), np.max(g2))
    max_b = np.max(b)

    # Assume g_scale is one
    r_scale = max_g / max_r
    b_scale = max_g / max_b

    # Apply scale,  Red(0, 0)
    I[0::2, 0::2] *= r_scale
    # Blue (1,1)
    I[1::2, 1::2] *= b_scale
    
    I /= np.max(I)
    return I

def gray_world_wb(raw_img):
    # Make a copy, so that the processing will not effect input 
    I =  raw_img.copy()
    # Extract the R, G1, G2, B data from the raw data
    r = I[0::2, 0::2]
    g1 = I[0::2, 1::2]
    g2 = I[1::2, 0::2]
    b = I[1::2, 1::2]
    
    # Average color assumed to be grey
    mean_r = np.mean(r)
    mean_g = (np.mean(g1) + np.mean(g2)) / 2
    mean_b = np.mean(b)
    
    # Assume g_scale is one
    r_scale = mean_g / mean_r
    g_scale = 1
    b_scale = mean_g / mean_b 
    
    # Apply scale,  Red(0, 0)
    I[0::2, 0::2] *= r_scale
    # Blue (1,1)
    I[1::2, 1::2] *= b_scale
    
    I /= np.max(I)
     
    return I

def camera_preset_wb(raw_img, r_scale, g_scale, b_scale):
    I =  raw_img.copy()
    # Red (0,0) 
    I[0::2, 0::2] *= r_scale
    # Blue (1,1)
    I[1::2, 1::2] *= b_scale
    # Green (1,0) and (0,1)
    I[1::2, 0::2] *= g_scale
    I[0::2, 1::2] *= g_scale  
    #print(np.min(I), np.max(I))
    I /= np.max(I)
    #print(np.min(I), np.max(I))    
    return I


def demosaicing(I):
    
    h, w = I.shape
    x = np.arange(w)
    y = np.arange(h)
          
    # Define interpolation functions for R,G, B channels
    interp_r = interp2d(x[0::2], y[0::2], I[0::2, 0::2], kind='linear', fill_value=0)
    interp_b = interp2d(x[1::2], y[1::2], I[1::2, 1::2], kind='linear', fill_value=0)
    interp_g1 = interp2d(x[0::2], y[1::2], I[0::2, 1::2], kind='linear', fill_value=0)
    interp_g2 = interp2d(x[1::2], y[0::2], I[1::2, 0::2], kind='linear', fill_value=0)
    
    
    # Interpolate missing pixels
    I_r = interp_r(x, y)
    I_b = interp_b(x, y)
    # Green Channel
    I_g = (interp_g1(x, y) + interp_g2(x, y)) / 2
    
    I_rgb = np.stack((I_r, I_g, I_b), axis=-1)
    #print(I_g.shape, I_rgb.shape)
    
    return I_rgb
    

#https://www.dechifro.org/dcraw/dcraw.c
#line 7608 under function  
#{ "Nikon D3", 0, 0,
#	{ 8139,-2171,-663,-8747,16541,2295,-1925,2008,8093 } },
def color_space_correction(I):

    # M sRGB -> XYZ
    M1 = np.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])
              
    # M XYZ-> cam 
    M2 = np.array([[8139,-2171,-663],
                   [-8747,16541,2295],
                   [-1925,2008,8093]], dtype=np.double) / 10000
                  
    # I is a HxWx3 3-D array, X is Hx by 3 2-D array 
    X = I.reshape(-1, 3)
    #M3: M sRGB->cam
    M3 = M2@M1
    #Normalize it so that its rows sum to 1.
    M3 = M3 / M3.sum(axis=1)[:, np.newaxis]  
    M3_inv = np.linalg.inv(M3)    
    #x_correct =  (M3_inv@x.T).T
    x_correct = X@M3_inv.T 
    x_correct = x_correct.reshape(I.shape)
    #print(x_correct.shape)
    return x_correct

def bright_adjustment(I, scale):
    I = I * scale
    x = rgb2gray(I)
    print(f'max{I.max():.4f}, min{I.min():.4f}')
    print(f'gray img mean:{np.mean(x)}')  
    y = np.clip(I, 0, 1)
    return y

def gamma_enc(x):
    # Define the threshold
    threshold = 0.0031308
    
    # Initialize the output array
    gamma_encoded = np.zeros_like(x)
    
    # Apply the gamma encoding formula to each channel
    mask = x <= threshold
    # Less than the threshold 
    gamma_encoded[mask] = 12.92 * x[mask]
    # Larger than the threshold 
    gamma_encoded[~mask] = (1.055 * np.power(x[~mask], 1/2.4)) - 0.055
  
    y = np.clip(gamma_encoded, 0, 1)
    
    return y
    