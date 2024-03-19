from skimage.metrics import structural_similarity as ssim
import cv2

# Load the images
image1 = cv2.imread('approximateresult.png')
image2 = cv2.imread('sequentialresult.png')

# Convert the images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
ssim_index = ssim(image1_gray, image2_gray)

# Print the SSIM index
print(f"The SSIM index between approximateresult.png and sequentialresult.png is: {ssim_index}")
