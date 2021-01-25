import face_recognition
import cv2

# 打开摄像头。读取摄像头拍到的画面

# 定位到画面中人的面部，并用绿色的框框 把人的脸部框起来


# 1打开摄像头
video_capture = cv2.VideoCapture(0)

# 2循环获取摄像头拍摄到的画面，做进一步处理

while True:
    # 2.1 获取摄像头拍摄到的画面
    ret, frame = video_capture.read()
    # 2.2 从拍摄画面中提取出人的脸部所在区域
    face_locations = face_recognition.face_locations(frame)
    # 2.3 循环遍历人到脸部所在区域，并画框
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 2.4 通过opencv把画面展示
    cv2.imshow("Video", frame)
    # 2.5 设置按q退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # 退出循环
        break

# 3推出程序的时候 释放摄像头或其他资源
video_capture.release()
cv2.destroyAllWindows()
