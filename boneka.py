import cv2
import numpy as np

# Load image
image1 = cv2.imread("g1.jpg")
image2 = cv2.imread("g2.jpg")
image3 = cv2.imread("g3.jpg")

# Convert BGR to HSV
def show_real_people(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the HSV range
    lower_range = np.array([0, 0, 0])
    upper_range = np.array([255, 180, 170])

    # Threshold the HSV image to get only colors in the specified range
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:4]

    # Draw bounding boxes around the top 3 largest contours
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"object {idx+1}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    object_detection = [
        cv2.matchShapes(contours[1],contours[2],1,0.0),
        cv2.matchShapes(contours[0],contours[1],1,0.0),
        cv2.matchShapes(contours[2],contours[0],1,0.0),
    ]

    real_people = object_detection.index(min(object_detection))

    cv2.putText(image, f"object {real_people+1} is real", (0,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
    image = cv2.resize(image, (720,480))

    # print("Matching Shape 2 with Shape 3:", object_detection[1])
    # print("Matching Shape 1 with Shape 2:", object_detection[0])
    # print("Matching Shape 3 with Shape 1:", object_detection[2])
    return image

# Display the resulting image
cv2.imshow("1", show_real_people(image1))
cv2.imshow("2", show_real_people(image2))
cv2.imshow("3", show_real_people(image3))
cv2.waitKey(0)
cv2.destroyAllWindows()
