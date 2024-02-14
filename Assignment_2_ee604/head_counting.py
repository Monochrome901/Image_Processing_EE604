import cv2
import numpy as np
def solution(image_path):
  image = cv2.imread(image_path)
  image = 255-image

  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
      if gray[i][j]>10:
        gray[i][j] = 255
      else:
        gray[i][j] = 0

  gray = cv2.medianBlur(gray,11)
  median = cv2.medianBlur(gray,5)

  image = cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)

  top_pad = 5
  bottom_pad = 5
  left_pad = 5
  right_pad = 5

  image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])


  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, 50, 150)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for contour in contours:
      min_rect = cv2.minAreaRect(contour)
      box = np.int0(cv2.boxPoints(min_rect))
      cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

  m, n = median.shape

  s = set()

  for i in range(m - 1, -1, -1):
      for j in range(n):
          if median[i, j] > 0:
              s.add(j)

      if len(s) != 0:
          break

  if len(s) != 0:
      z = (min(s) + max(s)) // 2 + 5
  z = float(z)

  x1 = box[0][0].astype(float)
  x2 = box[1][0].astype(float)
  x3 = box[2][0].astype(float)
  a = (x1,x2,x3)

  ratio = (z-a[0])/(a[2]-z)
  if ratio>0.86 and ratio<0.91:

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    maxi = 0
    idx = 0

    for i in range(gray.shape[0]):
      sum = 0
      for j in range(gray.shape[1]):
        if gray[i][j]<15:
          sum = sum+1
      if maxi<sum:
        maxi = sum
        idx = i

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      min_rect = cv2.minAreaRect(contour)
      box = np.int0(cv2.boxPoints(min_rect))
      cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
      x1 = box[0][0]
      x2 = box[1][0]
      x3 = box[2][0]
      a = (x1,x2,x3)

      fake = False

    for i in range(a[0],a[2]-20):
      sum = 0
      for k in range(idx-10,idx+10):
        for l in range(i-1,i+20):
          if gray[k][l]<10:
            sum = sum+1
      if sum == 0:
        fake = True
        break

    if fake == True:
      return "fake"
    else:
      return "real"
  else:
    return "fake"