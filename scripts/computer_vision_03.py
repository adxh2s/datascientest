import numpy as np
import cv2 #import OpenCV
import matplotlib.pyplot as plt

building_color = cv2.imread('/home/adxh2s/Projects/datascientest/data/img/building.png', cv2.IMREAD_COLOR)  
building_color = cv2.cvtColor(building_color, cv2.COLOR_BGR2RGB)

gaussian_filter = cv2.GaussianBlur(building_color,(3,3),0)
edges = cv2.Canny(gaussian_filter,100,200)

# Affichage
plt.imshow(edges, cmap = 'gray')                         
plt.xticks([])
plt.yticks([])                        
plt.show()  

lines = cv2.HoughLinesP(edges,rho=3,theta=np.pi / 20,threshold=100,minLineLength=0,maxLineGap= 20)

line_img = np.zeros((building_color.shape[0],building_color.shape[1],3), dtype=np.uint8)
    
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), color = [255, 0, 0], thickness=3)
        
building_lines = cv2.addWeighted(building_color, 0.8, line_img, 1.0, 0.0)

plt.figure(figsize = (8,5))
plt.imshow(building_lines)
plt.xticks([])
plt.yticks([])
plt.show()

street_gray = cv2.imread('/home/adxh2s/Projects/datascientest/data/img/street.png', cv2.IMREAD_GRAYSCALE)
sobel = cv2.Sobel(street_gray, ddepth = cv2.CV_64F, dx = 1, dy = 0)

plt.figure(figsize = (8,5))

plt.imshow(sobel, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

laplacian = cv2.Laplacian(street_gray, ddepth = cv2.CV_64F)

plt.figure(figsize = (8,5))

plt.imshow(laplacian, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

spider_gray = cv2.imread('/home/adxh2s/Projects/datascientest/data/img/spider.png', cv2.IMREAD_GRAYSCALE)
spider_gray = cv2.resize(spider_gray, (150, 150))
fig = plt.figure(figsize = (12,12))
plt.imshow(spider_gray, cmap = 'gray')
plt.xticks([])
plt.yticks([])

spider_gray = 255 - spider_gray
filtre = cv2.GaussianBlur(spider_gray, ksize = (3,3), sigmaX = 0)

fig = plt.figure(figsize = (12,12))

fig.add_subplot(1,4,1)

plt.imshow(spider_gray, cmap = 'gray')
plt.xticks([])
plt.yticks([])

fig.add_subplot(1,4,2)

kernel_1 = np.ones((3,3),np.uint8)
erosion_1 = cv2.erode(filtre, kernel_1)

plt.imshow(erosion_1, cmap = 'gray')
plt.xticks([])
plt.yticks([])

fig.add_subplot(1,4,3)

kernel_2 = np.ones((5,5),np.uint8)
erosion_2 = cv2.erode(filtre, kernel_2)

plt.imshow(erosion_2, cmap = 'gray')
plt.xticks([])
plt.yticks([])

fig.add_subplot(1,4,4)

kernel_3 = np.ones((7,7),np.uint8)
erosion_3 = cv2.erode(filtre, kernel_3)

plt.imshow(erosion_3, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

fig = plt.figure(figsize = (12,12))

fig.add_subplot(1,2,1)

kernel = np.ones((5,5),np.uint8)

dilatation = cv2.dilate(erosion_3,kernel)

plt.imshow(dilatation, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

diff = spider_gray - dilatation

plt.imshow(diff, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()