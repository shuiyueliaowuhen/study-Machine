import send_wx_notice as swn
import cv2
import datetime,time

camera = cv2.VideoCapture(0)
background = None
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 4))
last_time = 0

while True:
    ret, frame = camera.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (25, 25), 3)

    if background is None:
        background = gray_frame
        continue

    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=3)

    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    is_detected = False
    for c in contours:
        # 如果误差小于2k，忽略
        if cv2.contourArea(c) < 2000:
            continue
        # 获取误差的坐标
        (x, y, w, h) = cv2.boundingRect(c)
        # 画框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        is_detected = True
        # 当前时间戳 秒
        current_time = int(time.time())
        if current_time - last_time > 5:
            print(1)
            last_time = current_time
            # 根据微信公众号文档，看一下测试公众号怎么发的，send_wx_notice自己封装的方法
            swn.send_wx_notice("动了")

    if is_detected:
        show_text = 'Motion: Detected'
        show_color = (0, 0, 255)
    else:
        show_text = 'Motion: UnDetected'
        show_color = (0, 255, 0)

    cv2.putText(frame, show_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, show_color, 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, show_color, 1)
    cv2.imshow('video', frame)
    # cv2.imshow('diff', gray_frame)

    key = cv2.waitKey(1) & 0xFFf
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
