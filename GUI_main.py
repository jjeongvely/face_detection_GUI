from multiprocessing import Process, Queue

import sys
import datetime
import face_recognition

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import uic

import cv2

print("OpenCV version: {}".format(cv2.__version__))

from knn_face_recognition import *

form_class = uic.loadUiType("GUI.ui")[0]

class MainWindow(QWidget, form_class):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.setupUi(self)

        self.edit.setText("Input name")

        # create a timer
        # set timer timeout callback function

        # viewCam함수 즉, 화면 출력을 위한 timer
        self.view_timer = QTimer()
        self.view_timer.timeout.connect(self.viewCam)

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self.Detection)

        # save 버튼을 누르는지 보기위한 timer
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.saveName)

        # progressBar에 숫자를 표시해주기위한 timer
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.progress)

        self.controlTimer()
        self.list()

        # set detection callback clicked  function
        # face registration 버튼을 누르면 faceRegistration 함수 호출
        self.registration.clicked.connect(self.faceRegistration)
        # detection 버튼을 누르면 faceDetection 함수 호출
        self.detection.clicked.connect(self.faceDetection)

        # Setup process and queues for multiprocessing.
        self.save_queue = Queue()
        self.progress_queue = Queue()

        # 얼굴 등록시 화면에서 계속 영상이 나오도록 등록 process를 따로 만들어줌
        self.save_process = Process(target=train, args=(self.save_queue, self.progress_queue, "dataset", "trained_knn_model.clf", 2))
        self.save_process.start()


    # progress bar에 표시해줄 숫자를 받아옴
    def progress(self):
        num = self.progress_queue.get()
        if num == 10000:
            self.log_browser.setText("등록완료")
            self.progress_timer.stop()
        self.progressBar.setProperty("value", num)


    # 등록한 사람들의 이름 리스트를 보여줌
    def list(self):
        # 등록한 사람들의 data가 저장된 폴더
        dir = "/home/qisens/facedetection/dataset/"
        msg = ""
        for class_dir in os.listdir(dir):
            msg += str(class_dir + '\n')
            self.namelist.setText(msg)


    def faceDetection(self):
        # detect_timer가 stop상태이면 start시켜 detection 함수 호출
        if not self.detect_timer.isActive():
            self.log_browser.setText("[INFO] Start face detection")
            self.detect_timer.start(0.1)
            self.detection.setText("Stop")
        # detection 버튼을 다시 누르면 detect_timer가 start상태이므로 stop으로 변경
        else:
            self.detect_timer.stop()
            self.log_browser.setText("[INFO] Stop face detection")
            self.detection.setText("Start Detection")


    def Detection(self):
        # Grab a single frame of video
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 2)

        if ret is False:
            self.detect_timer.stop()
            self.log_browser.setText("[INFO] Detection is over...")
            self.detection.setText("Start Detection")

        # Resize frame of video to 1/4 size for faster face recognition processing
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        except Exception as e:
            print(str(e))

        # Find all the faces and face encodings in the current frame of video using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(rgb_small_frame, model_path="trained_knn_model.clf")

        # 화면의 사람을 예측하여 얼굴에 bounding box로 표시
        # Display results overlaid on an image
        for name, (top, right, bottom, left) in predictions:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # cv2에서는 BGR이므로 RGB로 바꿈
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width

        # create QImage from image
        qImg = QImage(frame, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        self.image_label.setPixmap(QPixmap(qImg))
        self.detect_timer.start(0.1)


    # 이름을 쓰고 save버튼을 누를때까지 timer작동
    def saveName(self):
        self.log_browser.setText("이름을 입력하세요")
        self.progressBar.setProperty("value", 0)
        success = True
        if self.save_bt.isDown() is True:
            self.log_browser.setText("잠시만 기다리세요")
            name = self.edit.text()
            self.save_timer.stop()
            # print(name)
            for i in range(10):
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 2)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                bounding_boxes = face_recognition.face_locations(frame)

                # 등록시 화면에 한명이 아닌경우
                if len(bounding_boxes) != 1:
                    self.log_browser.setText("등록실패! 한명만 시도하십시오")
                    success = False
                    break

                image_name = "{:%Y%m%dT%H%M%S}_{}.jpg".format(datetime.datetime.now(), i)
                folder_name = "/home/qisens/facedetection/dataset/"

                if not os.path.isdir(folder_name + name):
                    os.mkdir(folder_name + name)
                image.save(folder_name + name + '/' + image_name)

        # 등록에 성공한 경우에만 progress_timer start
        if success:
            self.progress_timer.start(0.1)
            # save_queue에 무언가를 넣어줌으로써 train함수 작동
            self.save_queue.put(1)
            self.list()


    # save_timer를 작동시킴
    def faceRegistration(self):
        self.save_timer.start(0.1)
        self.log_browser.setText("Start face registration")


    def closeEvent(self, event):
        self.save_process.terminate()
        self.save_process.join()
        self.view_timer.stop()
        self.detect_timer.stop()
        self.save_timer.stop()
        self.cap.release()


    # view camera
    def viewCam(self):
        ret, frame = self.cap.read() # 화면을 읽어오기 - 실패시 ret가 false

        if ret is False:
            self.view_timer.stop()
            self.cap.release()
            self.log_browser.setText("[INFO] video is over...")

        # If frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2) # 화면 반전

        # OpenCV에서는 BGR 순서로 저장하니까 BGR을 RGB로
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        # QLabel은 이미지를 표시하는 라벨 위젯
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        # create video capture
        print("[INFO] starting video stream thread...")
        self.log_browser.setText("[INFO] starting video stream thread...")
        self.cap = cv2.VideoCapture(-1) # cam 연결

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 화면 크기 설정

        # start timer
        self.view_timer.start(20) # view_timer start

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
