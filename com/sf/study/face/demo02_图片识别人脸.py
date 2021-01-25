import face_recognition
import cv2

img = cv2.imread("img/人脸测试.jpg")

face_locations = face_recognition.face_locations(img)

for top, right, bottom, left in face_locations:
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imwrite("img/人脸测试结果.png", img)
