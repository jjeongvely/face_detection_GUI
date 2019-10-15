from multiprocessing import Process, Queue

import sys
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time
import threading
import playsound
import datetime
import face_recognition
from face_recognition_cli import image_files_in_folder

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QStandardItemModel, QStandardItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5 import uic

# import Opencv module
import cv2
print("OpenCV version: {}".format(cv2.__version__))

from knn_face_recognition import *
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

form_class = uic.loadUiType("GUI.ui")[0]

# multiprocessing may not work on Windows and macOS, check OS for safety.
#detect_os()

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

facebox = None
ROI_region = None

# Icon for alarm
ICON_RED_LED = "./icons/led_circle_red.svg.med.png"
ICON_GREEN_LED = "./icons/led_circle_green.svg.med.png"
ICON_GREY_LED = "./icons/led_circle_grey.svg.med.png"

no_face_count = 0

def sound_alarm():
    global SOUND_ALARM
    playsound.playsound('./HyprBlip.wav')
    SOUND_ALARM = False

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)

# moving average filter
def moving_avg(values, value):
    if len(values) > 2 and isinstance(values, list):
        values.append(value)
        del values[0]
    else:
        values = list(values)
        values.append(value)
    value_avg = np.median(values, 0)  # selection np.average(values, 0)
    return value_avg, values

# calculate MAR
def mouth_aspect_ratio(mouth, t1, t2):
    global max_mouthLR
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])

    mouthLR = dist.euclidean(mouth[0], mouth[6])

    if max_mouthLR < mouthLR:
        max_mouthLR = mouthLR
    elif ALARM_ON_MOUTH:
        max_mouthLR = 0

    mouthUD = A+B+C / 3

    if mouthLR < max_mouthLR * .9:
        mar = mouthUD * t2 / mouthLR / t1

    else:
        mar = mouthUD * t2 / mouthLR

    return mar, mouthLR

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Video Open'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.openFileNameDialog()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Video Open", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)

class MainWindow(QWidget, form_class):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.setupUi(self)

        self.edit.setText("Input name")
        # create a timer
        self.view_timer = QTimer()
        # set timer timeout callback function
        self.view_timer.timeout.connect(self.viewCam)

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self.Detection)

        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self.saveName)

        self.controlTimer()
        self.list()
        # set detection callback clicked  function
        self.registration.clicked.connect(self.faceRegistration)
        self.detection.clicked.connect(self.faceDetection)


        # Introduce mark_detector to detect landmarks.
        self.mark_detector = MarkDetector()

        # Setup process and queues for multiprocessing.
        self.img_queue = Queue()
        self.box_queue = Queue()

        self.box_process = Process(target=get_face, args=(
            self.mark_detector, self.img_queue, self.box_queue,))
        self.box_process.start()

        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        self.pose_estimator = PoseEstimator(img_size=(480, 640))

        # Introduce scalar stabilizers for pose.
        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

    def list(self):
        dir = "/home/qisens/facedetection/dataset/"
        msg=""
        for class_dir in os.listdir(dir):
            msg+=str(class_dir+'\n')
            self.namelist.setText(msg)

    def faceDetection(self):
        if not self.detect_timer.isActive():
            self.log_browser.setText("[INFO] Start face detection")
            #print("detection")
            self.detect_timer.start(0.1)
            self.detection.setText("Stop Detection")
        else:
            self.detect_timer.stop()
            #print("no d")
            self.log_browser.setText("[INFO] Stop face detection")
            self.detection.setText("Start Detection")

    def Detection(self):
        # Grab a single frame of video
        #if not self.detect_timer.isActive():
        #print("start detection")
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 2)

        if ret is False:
            self.detect_timer.stop()
            self.log_browser.setText("[INFO] Detection is over...")
            self.detection.setText("Start Detection")
        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
        except Exception as e:
            print(str(e))

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time

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
            #print(name)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
        #print("end detection")
        self.detect_timer.start(0.1)
        #else:
            #print("stop detect_timer!!!!!!!!!!!")
            # stop timer
            #self.detect_timer.stop()
            # update control_bt text
            #self.log_browser.setText("[INFO] stop detection")

    def saveName(self):
        self.log_browser.setText("이름을 입력하세요")
        fail=False
        if self.save_bt.isDown() is True:
            name=self.edit.text()
            self.save_timer.stop()
            print(name)
            for i in range(10):
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 2)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                bounding_boxes=face_recognition.face_locations(frame)
                if len(bounding_boxes) != 1:
                    self.log_browser.setText("등록실패! 한명만 시도하십시오")
                    fail=False
                    break
                image_name = "{:%Y%m%dT%H%M%S}_{}.jpg".format(datetime.datetime.now(), i)
                folder_name = "/home/qisens/facedetection/dataset/"
                if not os.path.isdir(folder_name + name):
                    print("mkdir")
                    os.mkdir(folder_name + name)
                image.save(folder_name + name + '/' + image_name)
            fail=True
        if fail:
            self.log_browser.setText("잠시만 기다리세요")
            print("잠시만 기다리세요")
            self.progressBar.setProperty("value", 0)
            train(self.progressBar, "dataset", model_save_path="trained_knn_model.clf", n_neighbors=2)
            self.log_browser.setText("등록완료")

    def faceRegistration(self):
        self.save_timer.start(0.1)
        self.log_browser.setText("Start face registration")

    def closeEvent(self, event):
        self.box_process.terminate()
        self.box_process.join()
        self.view_timer.stop()
        self.cap.release()

    # view camera
    def viewCam(self):
        global buf_YAW, buf_PITCH, buf_MAR, buf_EAR, prev_time, EAR, MAR, YAW, PITCH, facebox,\
        ALARM_ON_EYE, ALARM_ON_face, ALARM_ON_MOUTH, ALARM_ON_pitch, ALARM_ON_yaw, no_face_count, SOUND_ALARM
        cur_time = time.time()
        sec = cur_time - prev_time
        prev_time = cur_time
        ret, frame = self.cap.read()

        if ret is False:
            self.view_timer.stop()
            self.cap.release()
            self.log_browser.setText("[INFO] video is over...")

        # If frame comes from webcam, flip it so it looks like a mirror.
        frame = cv2.flip(frame, 2)

        if ROI_region is not None:
            image_bgr_roi = filter_region(frame, ROI_region)
            cv2.rectangle(frame, (ROI_region[0][1][0], ROI_region[0][1][1]),
                          (ROI_region[0][3][0], ROI_region[0][3][1]), (255, 0, 0), 1)
            cv2.imshow('image_bgr_roi', image_bgr_roi)
        else:
            image_bgr_roi = frame

        # Feed frame to image queue.
        self.img_queue.put(image_bgr_roi)

        # Get face from box queue.
        facebox = self.box_queue.get()


        if facebox is None and EAR_calib != 0:
            self.alert_label_1.setPixmap(QPixmap(ICON_GREY_LED))
            self.alert_label_2.setPixmap(QPixmap(ICON_GREY_LED))
            self.alert_label_3.setPixmap(QPixmap(ICON_GREY_LED))
            self.alert_label_4.setPixmap(QPixmap(ICON_GREY_LED))
            self.alert_label_5.setPixmap(QPixmap(ICON_GREY_LED))
            no_face_count += 1
            if no_face_count > 15:
                no_face_count = 15

        """
        box_color = (0, 255, 0)

        if facebox is not None:
            ALARM_ON_face = False
            no_face_count = 0

            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            marks = self.mark_detector.detect_marks(face_img)

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            marks_int = marks.astype(int)

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))

            # Try pose estimation with 68 points.
            pose = self.pose_estimator.solve_pose_by_68_points(marks)
            # Stabilize the pose.
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            #[array([0.08098711], dtype=float32), array([0.19139864], dtype=float32), array([-3.106952], dtype=float32),
            # array([73.12916], dtype=float32), array([0.6036028], dtype=float32), array([-609.9196], dtype=float32)]

            stabile_pose = np.reshape(stabile_pose, (-1, 3))
            rvec = stabile_pose[0]

            YAW = rvec[0] * 57.2958
            PITCH = rvec[1] * 57.2958

            YAW = YAW - YAW_calib

            PITCH = PITCH - PITCH_calib


            t1 = np.abs(YAW) * 0.01 + 1
            t2 = np.abs(PITCH) * 0.01 +1

            leftEye = marks_int[lStart:lEnd]
            rightEye = marks_int[rStart:rEnd]
            mouth = marks_int[mStart:mEnd]
            nose = marks_int[nStart:nEnd]
            contour = marks_int[0:27]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)


            MAR, mouthLR = mouth_aspect_ratio(mouth, t1, t2)

            if mouthLR < max_mouthLR * 0.9:
                EAR = (leftEAR + rightEAR) * t2 / 2.0 / t1
            else:
                EAR = (leftEAR + rightEAR) * t2 / 2.0

            EAR = EAR - EAR_calib
            MAR = MAR - MAR_calib


            # moving avg filter
            EAR, buf_EAR = moving_avg(buf_EAR, EAR)
            MAR, buf_MAR = moving_avg(buf_MAR, MAR)
            YAW, buf_YAW = moving_avg(buf_YAW, YAW)
            PITCH, buf_PITCH = moving_avg(buf_PITCH, PITCH)

            save_frame = 15

            if EAR_calib != 0 and self.driving_bt.isChecked():
                if EAR < -self.ear_threshold_bar.value() / 100:
                    buf_EAR_judge.append(1)
                else:
                    buf_EAR_judge.append(0)

                if len(buf_EAR_judge) > save_frame:
                    buf_EAR_judge.pop(0)

                if MAR > self.mar_threshold_bar.value() / 10:
                    buf_MAR_judge.append(1)
                else:
                    buf_MAR_judge.append(0)

                if len(buf_MAR_judge) > save_frame:
                    buf_MAR_judge.pop(0)

                if abs(YAW) > self.yaw_threshold_bar.value():
                    buf_YAW_judge.append(1)
                else:
                    buf_YAW_judge.append(0)

                if len(buf_YAW_judge) > save_frame:
                    buf_YAW_judge.pop(0)

                if abs(PITCH) > self.pitch_threshold_bar.value():
                    buf_PITCH_judge.append(1)
                else:
                    buf_PITCH_judge.append(0)

                if len(buf_PITCH_judge) > save_frame:
                    buf_PITCH_judge.pop(0)

                if sum(buf_EAR_judge) >= self.drowsiness_threshold_bar.value() and self.driving_bt.isChecked():
                    self.alert_label_1.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_EYE = True
                    box_color = (0,0,255)
                else:
                    ALARM_ON_EYE = False

                if sum(buf_MAR_judge) >= self.yawn_threshold_bar.value() and self.driving_bt.isChecked():
                    self.alert_label_2.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_MOUTH = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_MOUTH = False

                if sum(buf_YAW_judge) >= self.carelessness_yaw_threshold_bar.value() and self.driving_bt.isChecked():
                    self.alert_label_3.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_yaw = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_yaw = False

                if sum(buf_PITCH_judge) >= self.carelessness_pitch_threshold_bar.value() and self.driving_bt.isChecked():
                    self.alert_label_4.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_pitch = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_pitch = False


            self.pose_estimator.draw_annotation_box(
                frame, stabile_pose[0], stabile_pose[1], color=box_color, line_width=5)

            
            if self.contour_checkbox.isChecked():
                for mark in contour:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)

            if self.eye_checkbox.isChecked():
                for mark in leftEye:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)
                for mark in rightEye:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)
            if self.mouth_checkbox.isChecked():
                for mark in mouth:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)

            if self.nose_checkbox.isChecked():
                for mark in nose:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)
        """

        if ALARM_ON_face or ALARM_ON_EYE or ALARM_ON_MOUTH or ALARM_ON_pitch or ALARM_ON_yaw:
            if not SOUND_ALARM:
                SOUND_ALARM = True
                t = threading.Thread(target = sound_alarm, args=())
                t.daemon = True
                t.start()

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
        # if timer is stopped
        #if not self.view_timer.isActive():
        # create video capture
        print("[INFO] starting video stream thread...")
        self.log_browser.setText("[INFO] starting video stream thread...")
        self.cap = cv2.VideoCapture(-1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # start timer
        self.view_timer.start(20)
        # if timer is started
        """
        else:
            # stop timer
            self.view_timer.stop()
            # release video capture
            self.cap.release()

            # update detection text
            self.log_browser.setText("[INFO] stopping video stream thread...")
            self.detection.setText("Face Detection")
        """


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
