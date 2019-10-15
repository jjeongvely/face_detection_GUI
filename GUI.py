# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(970, 640)
        self.detection = QtWidgets.QPushButton(Dialog)
        self.detection.setGeometry(QtCore.QRect(430, 530, 231, 31))
        self.detection.setObjectName("detection")
        self.image_label = QtWidgets.QLabel(Dialog)
        self.image_label.setGeometry(QtCore.QRect(10, 10, 651, 480))
        self.image_label.setFrameShape(QtWidgets.QFrame.Box)
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")
        self.log_browser = QtWidgets.QTextBrowser(Dialog)
        self.log_browser.setGeometry(QtCore.QRect(60, 570, 361, 31))
        self.log_browser.setObjectName("log_browser")
        self.registration = QtWidgets.QPushButton(Dialog)
        self.registration.setGeometry(QtCore.QRect(430, 570, 231, 31))
        self.registration.setObjectName("registration")
        self.edit = QtWidgets.QLineEdit(Dialog)
        self.edit.setGeometry(QtCore.QRect(700, 570, 231, 31))
        self.edit.setObjectName("edit")
        self.save_bt = QtWidgets.QPushButton(Dialog)
        self.save_bt.setGeometry(QtCore.QRect(700, 530, 231, 31))
        self.save_bt.setObjectName("save_bt")
        self.scrollArea = QtWidgets.QScrollArea(Dialog)
        self.scrollArea.setGeometry(QtCore.QRect(700, 10, 231, 481))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 229, 479))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.label.setGeometry(QtCore.QRect(10, 10, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.namelist = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.namelist.setGeometry(QtCore.QRect(10, 50, 211, 421))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.namelist.setFont(font)
        self.namelist.setText("")
        self.namelist.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.namelist.setObjectName("namelist")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_4)
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(60, 530, 361, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 550, 41, 31))
        self.label_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.image_label.raise_()
        self.detection.raise_()
        self.log_browser.raise_()
        self.registration.raise_()
        self.edit.raise_()
        self.save_bt.raise_()
        self.scrollArea.raise_()
        self.progressBar.raise_()
        self.label_2.raise_()

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Focusing Status Monitoring / Face Landmark Detection System"))
        self.detection.setText(_translate("Dialog", "Start Detection"))
        self.registration.setText(_translate("Dialog", "Face Registration"))
        self.save_bt.setText(_translate("Dialog", "Save"))
        self.label.setText(_translate("Dialog", "Name list"))
        self.label_2.setText(_translate("Dialog", "INFO"))
