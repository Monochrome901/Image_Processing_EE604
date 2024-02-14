import cv2
import numpy as np

# Usage
def solution(image_path):
  image = cv2.imread(image_path)
  # print(image.shape)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if image[i][j][0]>105 and image[i][j][1]>105 and image[i][j][2]>105:
        image[i][j][0] = 0
        image[i][j][1] = 0
        image[i][j][2] = 0
  # cv2_imshow(image)
  # Split the image into its color channels
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  gamma = 1.5
      # Perform the gamma transformation
  adjusted_image = np.power(image / 255.0, gamma) * 255.0
  adjusted_image = adjusted_image.astype(np.uint8)
  # cv2_imshow(adjusted_image)
  blue, green, red = cv2.split(adjusted_image)

  # Create an all-zero matrix for the blue channel
  blue_channel = blue.copy()
  blue_channel[:,:] = 0
  green_channel = green.copy()
  green_channel[:,:] = 0
  # Merge the red and green channels to create the final image
  red_green_image = cv2.merge((blue_channel,green_channel,red))

  # Display or save the red and green channel image
  # cv2_imshow(red_green_image)

  gray = cv2.cvtColor(red_green_image,cv2.COLOR_BGR2GRAY)
  # cv2_imshow(gray)
  kernel_size = (5,5)
  alpha = 1.5
  blurred_image = cv2.GaussianBlur(red_green_image, kernel_size, 0)

  restored_image = cv2.addWeighted(red_green_image, alpha, blurred_image, -alpha, 0)
  gray = cv2.cvtColor(red_green_image, cv2.COLOR_BGR2GRAY)

  # Apply GaussianBlur to reduce noise and improve circle detection
  blurred = cv2.GaussianBlur(gray, (9, 9), 2)

  # Apply Hough Circle Transform
  circles = cv2.HoughCircles(
      blurred, 
      cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=80
  )

  if circles is not None:
    image = cv2.subtract(image,image)
    return image

  gray_image = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2GRAY)

  _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # cv2_imshow(thresh)

  # Apply the cross-median filter
  kernel = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)
  filtered_image = cv2.medianBlur(thresh, ksize=25)
  # Initialize an empty set
  s = set()

  # Assuming you have a 2D array 'arr' (a list of lists) with elements as strings
  # and 'm' and 'n' are the dimensions of the array
  # cv2_imshow(filtered_image)
  for i in range(filtered_image.shape[0]):
      for j in range(filtered_image.shape[1]):
          if filtered_image[i][j] == 255:
              s.add(j)

      if len(s) != 0:
          x = min(s)
          y = max(s)
          for j in range(x, y + 1):
              filtered_image[i][j] = 255
      # Clear the set for the next row
      s.clear()
  filtered_image = cv2.cvtColor(filtered_image,cv2.COLOR_GRAY2BGR)
  return filtered_image
