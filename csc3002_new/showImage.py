import cv2

# Load an image
image = cv2.imread('build/result.png')
image2 = cv2.imread("forest.jpg")
print(image.shape, " Dehazed Image Dimensions")
print(image2.shape, " Original Image Dimensions")
cv2.imshow("Original Image", image2)
cv2.imshow('Dehazed Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
