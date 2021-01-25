import os

import face_recognition
import cv2

# 打开摄像头。读取摄像头拍到的画面
# 定位到画面中人的面部，并用绿色的框框 把人的脸部框起来
# 拍摄到的人和数据库的人 面部面部特征比对，并在绿框上显示人命，不认识的显示unknown

# 读取数据库中的人名和面部特征
face_databases_dir = 'face_databases'
# 用户姓名
user_names = []
# 存用户面部特征向量
user_faces_encodings = []

# 读取文件夹
files = os.listdir(face_databases_dir)
# 循环读取文件
for img_shot_name in files:
    # 截取文件名，放入user_names
    user_name, _ = os.path.splitext(img_shot_name)
    user_names.append(user_name)
    # 读取面部特征，存储user_face
    image_file_name = os.path.join(face_databases_dir, img_shot_name)
    image_file = face_recognition.load_image_file(image_file_name)
    face_encoding = face_recognition.face_encodings(image_file)[0]
    user_faces_encodings.append(face_encoding)

# 1打开摄像头
video_capture = cv2.VideoCapture(0)

# 2循环获取摄像头拍摄到的画面，做进一步处理
while True:
    # 2.1 获取摄像头拍摄到的画面
    ret, frame = video_capture.read()
    # 2.2 从拍摄画面中提取出人的脸部所在区域
    face_locations = face_recognition.face_locations(frame)
    # 2.2.1从所有人的头像所在区域提取脸部特征
    current_face_encodings = face_recognition.face_encodings(frame, face_locations)
    # 2.2用于存储拍摄到用户的姓名列表
    names = []
    # 遍历current_face_encodings，和数据库做匹配
    for face_encoding in current_face_encodings:
        # user_faces_encodings 数据库的面部特征列表，face_encoding当前
        # 返回结果[true,false,false]
        matchs = face_recognition.compare_faces(user_faces_encodings, face_encoding)
        name = "UnKnown"
        for index, is_match in enumerate(matchs):
            if is_match:
                name = user_names[index]
                break
        names.append(name)

    # 2.3 循环遍历人到脸部所在区域，并画框 在框上面标示人名
    # zip
    for (top, right, bottom, left), name in zip(face_locations, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 字体
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, top-10), font, 0.5, (0, 255, 0), 1)

    # 2.4 通过opencv把画面展示
    cv2.imshow("Video", frame)
    # 2.5 设置按q退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # 退出循环
        break

# 3推出程序的时候 释放摄像头或其他资源
video_capture.release()
cv2.destroyAllWindows()
