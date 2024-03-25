from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import cv2
import numpy as np

# Load the images
image1 = cv2.imread('approximateresult.png')
image2 = cv2.imread('forest.jpg')
image3 = cv2.imread('sequentialresult.png')

# Convert the images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
ssim_index1 = ssim(image1_gray, image2_gray)
ssim_index2 = ssim(image3_gray, image2_gray)

# Compute PSNR between two images
psnr_index1 = psnr(image1_gray, image2_gray)
psnr_index2 = psnr(image3_gray, image2_gray)

# Compute MSE between two images
mse_index1 = mse(image1_gray, image2_gray)
mse_index2 = mse(image3_gray, image2_gray)

# Print the SSIM, PSNR, and MSE indices
print(f"The results for approximateresult.png and hills.jpg are -\nSSIM: {ssim_index1}\nPSNR: {psnr_index1}\nMSE: {mse_index1}\n")
print(f"\nThe results for sequentialresult.png and hills.jpg are -\nSSIM: {ssim_index2}\nPSNR: {psnr_index2}\nMSE: {mse_index2}\n")

cv2.imshow("Original Image" , image2_gray)
cv2.imshow("Dehazed Image" , image3_gray)
cv2.waitKey(0)
cv2.destroyAllWindows() 


