from multiprocessing import Process, Queue

import numpy as np
import sys
from imutils import face_utils
from scipy.spatial import distance as dist
import time
import threading
import playsound

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon

# import Opencv module
import cv2
print("OpenCV version: {}".format(cv2.__version__))

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

from FSMS_GUI import *

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


def calibration_button_clicked():
    global EAR_calib, MAR_calib, YAW_calib, PITCH_calib, ROI_region
    EAR_calib = EAR_calib + EAR
    MAR_calib = MAR_calib + MAR
    YAW_calib = YAW_calib + YAW
    PITCH_calib = PITCH_calib + PITCH

    roi_size = 75
    if facebox is not None:
        bottom_left = [facebox[0]-roi_size,facebox[3]+roi_size]
        top_left = [facebox[0]-roi_size, facebox[1]-roi_size]
        top_right = [facebox[2]+roi_size, facebox[1]-roi_size]
        bottom_right = [facebox[2]+roi_size, facebox[3]+roi_size]


        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        ROI_region = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    else:
        ROI_region = None

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

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

        self.ui.calibration_button.clicked.connect(calibration_button_clicked)

        # threshold set value
        self.ui.ear_threshold_browser.setText('%.2f' %(-self.ui.ear_threshold_bar.value()/100))
        self.ui.mar_threshold_browser.setText('%.2f' % (self.ui.mar_threshold_bar.value() / 10))
        self.ui.yaw_threshold_browser.setText('%d' % (self.ui.yaw_threshold_bar.value()))
        self.ui.pitch_threshold_browser.setText('%d' % (self.ui.pitch_threshold_bar.value()))
        self.ui.drowsiness_threshold_browser.setText('%d' % (self.ui.drowsiness_threshold_bar.value()))
        self.ui.yawn_threshold_browser.setText('%d' % (self.ui.yawn_threshold_bar.value()))
        self.ui.carelessness_yaw_threshold_browser.setText('%d' % (self.ui.carelessness_yaw_threshold_bar.value()))
        self.ui.carelessness_pitch_threshold_browser.setText('%d' % (self.ui.carelessness_pitch_threshold_bar.value()))
        self.ui.face_threshold_browser.setText('%d' %(self.ui.face_threshold_bar.value()))

        # threshold value changed
        self.ui.ear_threshold_bar.valueChanged.connect(self.ear_threshold_changed)
        self.ui.mar_threshold_bar.valueChanged.connect(self.mar_threshold_changed)
        self.ui.yaw_threshold_bar.valueChanged.connect(self.yaw_threshold_changed)
        self.ui.pitch_threshold_bar.valueChanged.connect(self.pitch_threshold_changed)
        self.ui.drowsiness_threshold_bar.valueChanged.connect(self.drowsiness_threshold_changed)
        self.ui.yawn_threshold_bar.valueChanged.connect(self.yawn_threshold_changed)
        self.ui.carelessness_yaw_threshold_bar.valueChanged.connect(self.carelessness_yaw_threshold_changed)
        self.ui.carelessness_pitch_threshold_bar.valueChanged.connect(self.carelessness_pitch_threshold_changed)
        self.ui.face_threshold_bar.valueChanged.connect(self.face_threshold_changed)

        self.ui.alert_label_1.setPixmap(QPixmap(ICON_GREY_LED))
        self.ui.alert_label_2.setPixmap(QPixmap(ICON_GREY_LED))
        self.ui.alert_label_3.setPixmap(QPixmap(ICON_GREY_LED))
        self.ui.alert_label_4.setPixmap(QPixmap(ICON_GREY_LED))
        self.ui.alert_label_5.setPixmap(QPixmap(ICON_GREY_LED))


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

    def closeEvent(self, event):
        self.box_process.terminate()
        self.box_process.join()
        self.timer.stop()
        self.cap.release()

    def ear_threshold_changed(self):
        self.ui.ear_threshold_browser.setText('%.2f' % (-self.ui.ear_threshold_bar.value() / 100))

    def mar_threshold_changed(self):
        self.ui.mar_threshold_browser.setText('%.2f' % (self.ui.mar_threshold_bar.value() / 10))

    def yaw_threshold_changed(self):
        self.ui.yaw_threshold_browser.setText('%d' % (self.ui.yaw_threshold_bar.value()))

    def pitch_threshold_changed(self):
        self.ui.pitch_threshold_browser.setText('%d' % (self.ui.pitch_threshold_bar.value()))

    def drowsiness_threshold_changed(self):
        self.ui.drowsiness_threshold_browser.setText('%d' % (self.ui.drowsiness_threshold_bar.value()))

    def yawn_threshold_changed(self):
        self.ui.yawn_threshold_browser.setText('%d' % (self.ui.yawn_threshold_bar.value()))

    def carelessness_yaw_threshold_changed(self):
        self.ui.carelessness_yaw_threshold_browser.setText('%d' % (self.ui.carelessness_yaw_threshold_bar.value()))

    def carelessness_pitch_threshold_changed(self):
        self.ui.carelessness_pitch_threshold_browser.setText('%d' % (self.ui.carelessness_pitch_threshold_bar.value()))

    def face_threshold_changed(self):
        self.ui.face_threshold_browser.setText('%d' % (self.ui.face_threshold_bar.value()))

    # view camera
    def viewCam(self):
        global buf_YAW, buf_PITCH, buf_MAR, buf_EAR, prev_time, EAR, MAR, YAW, PITCH, facebox,\
        ALARM_ON_EYE, ALARM_ON_face, ALARM_ON_MOUTH, ALARM_ON_pitch, ALARM_ON_yaw, no_face_count, SOUND_ALARM
        cur_time = time.time()
        sec = cur_time - prev_time
        prev_time = cur_time
        ret, frame = self.cap.read()


        if ret is False:
            self.timer.stop()
            self.cap.release()
            self.ui.log_browser.setText("[INFO] video is over...")
            self.ui.control_bt.setText("Start Cam/Video")

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
            self.ui.alert_label_1.setPixmap(QPixmap(ICON_GREY_LED))
            self.ui.alert_label_2.setPixmap(QPixmap(ICON_GREY_LED))
            self.ui.alert_label_3.setPixmap(QPixmap(ICON_GREY_LED))
            self.ui.alert_label_4.setPixmap(QPixmap(ICON_GREY_LED))
            self.ui.alert_label_5.setPixmap(QPixmap(ICON_GREY_LED))
            no_face_count += 1
            if no_face_count > 15:
                no_face_count = 15

        box_color = (0, 255, 0)
        if no_face_count > self.ui.face_threshold_bar.value() and self.ui.driving_bt.isChecked():
            ALARM_ON_face = True
            self.ui.alert_label_5.setPixmap(QPixmap(ICON_RED_LED))

        if facebox is not None:
            ALARM_ON_face = False
            no_face_count = 0
            if self.ui.driving_bt.isChecked() and EAR_calib != 0:
                self.ui.alert_label_1.setPixmap(QPixmap(ICON_GREEN_LED))
                self.ui.alert_label_2.setPixmap(QPixmap(ICON_GREEN_LED))
                self.ui.alert_label_3.setPixmap(QPixmap(ICON_GREEN_LED))
                self.ui.alert_label_4.setPixmap(QPixmap(ICON_GREEN_LED))
                self.ui.alert_label_5.setPixmap(QPixmap(ICON_GREEN_LED))
            if self.ui.no_driving_bt.isChecked():
                self.ui.alert_label_1.setPixmap(QPixmap(ICON_GREY_LED))
                self.ui.alert_label_2.setPixmap(QPixmap(ICON_GREY_LED))
                self.ui.alert_label_3.setPixmap(QPixmap(ICON_GREY_LED))
                self.ui.alert_label_4.setPixmap(QPixmap(ICON_GREY_LED))
                self.ui.alert_label_5.setPixmap(QPixmap(ICON_GREY_LED))
            landmarks_start = time.time()
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

            if EAR_calib != 0 and self.ui.driving_bt.isChecked():
                if EAR < -self.ui.ear_threshold_bar.value() / 100:
                    buf_EAR_judge.append(1)
                else:
                    buf_EAR_judge.append(0)

                if len(buf_EAR_judge) > save_frame:
                    buf_EAR_judge.pop(0)

                if MAR > self.ui.mar_threshold_bar.value() / 10:
                    buf_MAR_judge.append(1)
                else:
                    buf_MAR_judge.append(0)

                if len(buf_MAR_judge) > save_frame:
                    buf_MAR_judge.pop(0)

                if abs(YAW) > self.ui.yaw_threshold_bar.value():
                    buf_YAW_judge.append(1)
                else:
                    buf_YAW_judge.append(0)

                if len(buf_YAW_judge) > save_frame:
                    buf_YAW_judge.pop(0)

                if abs(PITCH) > self.ui.pitch_threshold_bar.value():
                    buf_PITCH_judge.append(1)
                else:
                    buf_PITCH_judge.append(0)

                if len(buf_PITCH_judge) > save_frame:
                    buf_PITCH_judge.pop(0)

                if sum(buf_EAR_judge) >= self.ui.drowsiness_threshold_bar.value() and self.ui.driving_bt.isChecked():
                    self.ui.alert_label_1.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_EYE = True
                    box_color = (0,0,255)
                else:
                    ALARM_ON_EYE = False

                if sum(buf_MAR_judge) >= self.ui.yawn_threshold_bar.value() and self.ui.driving_bt.isChecked():
                    self.ui.alert_label_2.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_MOUTH = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_MOUTH = False

                if sum(buf_YAW_judge) >= self.ui.carelessness_yaw_threshold_bar.value() and self.ui.driving_bt.isChecked():
                    self.ui.alert_label_3.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_yaw = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_yaw = False

                if sum(buf_PITCH_judge) >= self.ui.carelessness_pitch_threshold_bar.value() and self.ui.driving_bt.isChecked():
                    self.ui.alert_label_4.setPixmap(QPixmap(ICON_RED_LED))
                    ALARM_ON_pitch = True
                    box_color = (0, 0, 255)
                else:
                    ALARM_ON_pitch = False


            # ui set value
            self.ui.ear_browser.setText("EAR : %.3f" % (EAR))
            self.ui.mar_browser.setText("MAR : %.3f" % (MAR))
            self.ui.yaw_browser.setText("yaw : %.3f" % (YAW))
            self.ui.pitch_browser.setText("pitch : %.3f" % (PITCH))

            if EAR_calib != 0:
                self.ui.ear_alert_browser.setText("%d f" % (sum(buf_EAR_judge)))
                self.ui.mar_alert_browser.setText("%d f" % (sum(buf_MAR_judge)))
                self.ui.yaw_alert_browser.setText("%d f" % (sum(buf_YAW_judge)))
                self.ui.pitch_alert_browser.setText("%d f" % (sum(buf_PITCH_judge)))

            self.pose_estimator.draw_annotation_box(
                frame, stabile_pose[0], stabile_pose[1], color=box_color, line_width=5)
            if self.ui.contour_checkbox.isChecked():
                for mark in contour:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)

            if self.ui.eye_checkbox.isChecked():
                for mark in leftEye:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)
                for mark in rightEye:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)
            if self.ui.mouth_checkbox.isChecked():
                for mark in mouth:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)

            if self.ui.nose_checkbox.isChecked():
                for mark in nose:
                    cv2.circle(frame, (mark[0],
                                       mark[1]), 1, (0, 0, 255), -1, cv2.LINE_AA)

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
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

        self.ui.time_browser.setText("%.3f sec" % (sec))



    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            print("[INFO] starting video stream thread...")
            self.ui.log_browser.setText("[INFO] starting video stream thread...")
            if self.ui.webcam_bt.isChecked():
                webcam_src = self.ui.webcam_src_box.value()

                self.cap = cv2.VideoCapture(webcam_src)

            elif self.ui.video_bt.isChecked():
                self.app = App()
                self.app.show()
                self.cap = cv2.VideoCapture(self.app.fileName)
                self.app.close()

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop Cam/Video")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()

            # update control_bt text
            self.ui.log_browser.setText("[INFO] stopping video stream thread...")
            self.ui.control_bt.setText("Start Cam/Video")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
