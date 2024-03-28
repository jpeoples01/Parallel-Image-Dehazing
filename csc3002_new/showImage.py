import cv2

# Load an image
image = cv2.imread('approximateresult.png')
image2 = cv2.imread('sequentialresult.png')
image3 = cv2.imread("forest.jpg")
image4 = cv2.imread("halfprecisionresult.png")

# Define the new size
new_size = (500, 500)

# Resize the images
image = cv2.resize(image, new_size)
image2 = cv2.resize(image2, new_size)
image3 = cv2.resize(image3, new_size)
image4 = cv2.resize(image4, new_size)

# Display the images
cv2.imshow("Original Image", image3)
cv2.imshow('Approximate Dehazed Image', image)
cv2.imshow('Sequential Dehazed Image', image2)
cv2.imshow('Half Precision Dehazed Image', image4)
cv2.waitKey(0)
cv2.destroyAllWindows() 

