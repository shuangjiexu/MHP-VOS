from __future__ import division

import numpy as np
import scipy.ndimage
import cv2
from numba import jit


@jit
def computeAlphaJit(alpha, b, unknown):
    h, w = unknown.shape
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4  * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - b[i,j])
        n +=1
    return alphaNew


def poisson_matte(gray_img, trimap):
    h, w = gray_img.shape
    fg = trimap == 255
    bg = trimap == 0
    unknown = True ^ np.logical_or(fg,bg)
    fg_img = gray_img*fg
    bg_img = gray_img*bg
    alphaEstimate = fg + 0.5 * unknown

    approx_bg = cv2.inpaint(bg_img.astype(np.uint8),(unknown+fg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(fg)).astype(np.float32)
    approx_fg = cv2.inpaint(fg_img.astype(np.uint8),(unknown+bg).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)*(np.logical_not(bg)).astype(np.float32)

    # Smooth F - B image
    approx_diff = approx_fg - approx_bg
    approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)

    dy, dx = np.gradient(gray_img)
    d2y, _ = np.gradient(dy/approx_diff)
    _, d2x = np.gradient(dx/approx_diff)
    b = d2y + d2x

    alpha = computeAlphaJit(alphaEstimate, b, unknown)
    
    alpha = np.minimum(np.maximum(alpha,0),1).reshape(h,w)
    return alpha

# Load in image
def main():    
    img = scipy.misc.imread('troll.png')
    gray_img = scipy.misc.imread('troll.png', flatten='True')
    trimap = scipy.misc.imread('trollTrimap.bmp', flatten='True')

    alpha = poisson_matte(gray_img,trimap)

    plt.imshow(alpha, cmap='gray')
    plt.show()
    h, w, c = img.shape
    plt.imshow((alpha.reshape(h,w,1).repeat(3,2)*img).astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()