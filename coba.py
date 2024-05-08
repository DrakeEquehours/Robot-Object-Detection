import cv2

min_blue = 97
min_green = 88
min_red = 76
max_blue = 219
max_green = 255
max_red = 255
min_area_threshold = 400  # Adjust this value as per your requirement

frame = cv2.imread("g1.jpg")

# converting image into HSV color space as opencv reads image in BGR color model by default
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# getting the mask image from the HSV image using threshold values
mask = cv2.inRange(hsv_frame, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
# extracting the contours of the object
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# sorting the contour based of area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for contour in contours:
# calculating area for each contour
    area = cv2.contourArea(contour)
if area > min_area_threshold:
    # getting bounding box for contour if area exceeds threshold
    x_min, y_min, box_width, box_height = cv2.boundingRect(contour)
    # drawing a rectangle around the object with 15 as margin
    cv2.rectangle(frame, (x_min - 15, y_min - 15),
                    (x_min + box_width + 15, y_min + box_height + 15),
                    (0, 255, 0), 4)

# showing each frame of the video
cv2.imshow('frame', frame)
cv2.imshow('Mask Image', mask)
