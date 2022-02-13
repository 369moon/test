import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

#
#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges, cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()

# time speed show compare


img = cv2.imread("messi5.jpg", 0) # 0 gray, 3 color

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft) 

#magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])) 
#magnitude_spectrum = 20*np.log(np.abs(fshift)) 

rows, cols = img.shape
crow, ccol = rows//2, cols//2 #integer, / real

mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.subplot(1,2,1),plt.imshow(img, cmap='grey')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_back, cmap='grey')
plt.subplot(122),plt.imshow(img_back,cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()


#edges = cv2.Canny(img,100,200)

#cv2.waitKey()

#cv2.destroyAllWindows()