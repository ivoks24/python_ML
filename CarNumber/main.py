import cv2
import imutils
import numpy as np
import pytesseract


img = cv2.imread('carChina.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (600, 400))


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

edged = cv2.Canny(gray, 30, 200)
# cv2.imshow("sd", edged)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # im = cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
    # cv2.imshow("", im)
    # cv2.waitKey(0)

    peri = cv2.arcLength(box, True)
    # approx = cv2.approxPolyDP(box, 0.018 * peri, True)

    if peri > 300:
        screenCnt = box
        # im = cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
        # cv2.imshow("", im)
        # cv2.waitKey(0)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        Cropped_res = cv2.resize(Cropped, (120, 60))

        text = pytesseract.image_to_string(Cropped_res, config='--psm 11')

        if len(text) > 4:
            print("Detected license plate Number is:", text)
            im = cv2.drawContours(img, [box], -1, (0, 255, 0), 1)

            img = cv2.resize(img, (500, 300))
            Cropped = cv2.resize(Cropped, (400, 200))
            cv2.imshow('Cropped', Cropped)
            # cv2.imshow('Cropped', im)
            cv2.waitKey(0)

cv2.imshow('car', img)

cv2.waitKey(0)
cv2.destroyAllWindows()