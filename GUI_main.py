from multiprocessing import Process, Queue

import sys
from imutils import face_utils
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

CNN_INPUT_SIZE = 128

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# Alarm
ALARM_ON_EYE = False
ALARM_ON_MOUTH = False
ALARM_ON_yaw = False
ALARM_ON_pitch = False
ALARM_ON_face = False

SOUND_ALARM = False

max_mouthLR = 0

# buffer for judgement
buf_YAW = []
buf_PITCH = []
buf_MAR = []
buf_EAR = []
buf_EAR_judge = []
buf_MAR_judge = []
buf_YAW_judge = []
buf_PITCH_judge = []

prev_time = 0

# variables for judgement
EAR = 0
MAR = 0
yaw = 0
pitch = 0
EAR_calib = 0
MAR_calib = 0
YAW_calib = 0
PITCH_calib = 0

# Icon for alarm
ICON_RED_LED = "./icons/led_circle_red.svg.med.png"
ICON_GREEN_LED = "./icons/led_circle_green.svg.med.png"
ICON_GREY_LED = "./icons/led_circle_grey.svg.med.png"

no_face_count = 0


class MainWindow(QWidget, form_class):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.setupUi(self)

        self.edit.setText("Input name")

        # create a timer
        self.view_timer = QTimer() # 화면 출력을 위한 timer
        # set timer timeout callback function
        self.view_timer.timeout.connect(self.viewCam)

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self.Detection)

        self.save_timer = QTimer() # save 버튼을 누르는지 보기위한 timer
        self.save_timer.timeout.connect(self.saveName)

        self.progress_timer = QTimer() # progressBar에 숫자를 표시해주기위한 timer
        self.progress_timer.timeout.connect(self.progress)

        self.controlTimer()
        self.list()

        # set detection callback clicked  function
        self.registration.clicked.connect(self.faceRegistration)
        self.detection.clicked.connect(self.faceDetection)

        # Introduce mark_detector to detect landmarks.
        # self.mark_detector = MarkDetector()

        # Setup process and queues for multiprocessing.
        self.save_queue = Queue()
        self.progress_queue = Queue()

        self.save_process = Process(target=train, args=(self.save_queue, self.progress_queue, "dataset", "trained_knn_model.clf", 2))
        self.save_process.start()


    def progress(self):
        num = self.progress_queue.get()
        if num == 10000:
            self.log_browser.setText("등록완료")
            self.progress_timer.stop()
        self.progressBar.setProperty("value", num)


    def list(self):
        dir = "/home/qisens/facedetection/dataset/"
        msg = ""
        for class_dir in os.listdir(dir):
            msg += str(class_dir + '\n')
            self.namelist.setText(msg)


    def faceDetection(self):
        if not self.detect_timer.isActive():
            self.log_browser.setText("[INFO] Start face detection")
            # print("detection")
            self.detect_timer.start(0.1)
            self.detection.setText("Stop Detection")
        else:
            self.detect_timer.stop()
            # print("no d")
            self.log_browser.setText("[INFO] Stop face detection")
            self.detection.setText("Start Detection")


    def Detection(self):
        # Grab a single frame of video
        # if not self.detect_timer.isActive():
        # print("start detection")
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
            # print(name)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
        self.detect_timer.start(0.1)


    def saveName(self):
        self.log_browser.setText("이름을 입력하세요")
        self.progressBar.setProperty("value", 0)
        fail = False
        if self.save_bt.isDown() is True:
            name = self.edit.text()
            self.save_timer.stop()
            # print(name)
            for i in range(10):
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 2)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                bounding_boxes = face_recognition.face_locations(frame)
                if len(bounding_boxes) != 1:
                    self.log_browser.setText("등록실패! 한명만 시도하십시오")
                    fail = False
                    break
                image_name = "{:%Y%m%dT%H%M%S}_{}.jpg".format(datetime.datetime.now(), i)
                folder_name = "/home/qisens/facedetection/dataset/"
                if not os.path.isdir(folder_name + name):
                    print("mkdir")
                    os.mkdir(folder_name + name)
                image.save(folder_name + name + '/' + image_name)
            fail = True
        if fail:
            self.log_browser.setText("잠시만 기다리세요")
            print("잠시만 기다리세요")
            self.progress_timer.start(0.1)
            self.save_queue.put(1)
            self.list()


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
        ret, frame = self.cap.read()

        if ret is False:
            self.view_timer.stop()
            self.cap.release()
            self.log_browser.setText("[INFO] video is over...")

        # If frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2) # 화면 반전

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
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
