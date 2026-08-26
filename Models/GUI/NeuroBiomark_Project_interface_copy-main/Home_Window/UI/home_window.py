# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'home_window.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QProgressBar, QSizePolicy, QSlider,
    QTextEdit, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1533, 800)
        MainWindow.setMinimumSize(QSize(0, 0))
        MainWindow.setMaximumSize(QSize(11650000, 11650000))
        MainWindow.setStyleSheet(u"QMainWindow {\n"
"    background-color: #182034;\n"
"}")
        self.a_browse_image = QAction(MainWindow)
        self.a_browse_image.setObjectName(u"a_browse_image")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.a_browse_image.setFont(font)
        self.a_run_analysis = QAction(MainWindow)
        self.a_run_analysis.setObjectName(u"a_run_analysis")
        self.a_run_analysis.setFont(font)
        self.a_export_results = QAction(MainWindow)
        self.a_export_results.setObjectName(u"a_export_results")
        self.a_export_results.setFont(font)
        self.a_clear = QAction(MainWindow)
        self.a_clear.setObjectName(u"a_clear")
        self.a_clear.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_6 = QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMinimumSize(QSize(650, 300))
        self.groupBox_3.setMaximumSize(QSize(650, 300))
        self.groupBox_3.setStyleSheet(u"QGroupBox {\n"
"    background-color: rgba(42, 56, 82, 180);\n"
"    border-radius: 12px;\n"
"    border: none;\n"
"}")
        self.gridLayout_4 = QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_33 = QLabel(self.groupBox_3)
        self.label_33.setObjectName(u"label_33")
        self.label_33.setMinimumSize(QSize(200, 30))
        self.label_33.setMaximumSize(QSize(200, 30))
        self.label_33.setFont(font)
        self.label_33.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_33.setFrameShape(QFrame.NoFrame)
        self.label_33.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_33, 1, 1, 1, 1)

        self.label_43 = QLabel(self.groupBox_3)
        self.label_43.setObjectName(u"label_43")
        self.label_43.setMinimumSize(QSize(200, 30))
        self.label_43.setMaximumSize(QSize(16777215, 30))
        self.label_43.setFont(font)
        self.label_43.setStyleSheet(u"QLabel {\n"
"    background-color: #309090;\n"
"    color: #F5F5F5;\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.label_43.setFrameShape(QFrame.NoFrame)
        self.label_43.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_43, 0, 0, 1, 3)

        self.lb_input_image = QLabel(self.groupBox_3)
        self.lb_input_image.setObjectName(u"lb_input_image")
        self.lb_input_image.setMinimumSize(QSize(200, 200))
        self.lb_input_image.setMaximumSize(QSize(200, 200))
        self.lb_input_image.setStyleSheet(u"QLabel {\n"
"    background-color: #dfd7cf;\n"
"}")
        self.lb_input_image.setFrameShape(QFrame.WinPanel)
        self.lb_input_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.lb_input_image, 3, 0, 1, 1)

        self.lb_qu_path_image = QLabel(self.groupBox_3)
        self.lb_qu_path_image.setObjectName(u"lb_qu_path_image")
        self.lb_qu_path_image.setMinimumSize(QSize(200, 200))
        self.lb_qu_path_image.setMaximumSize(QSize(200, 200))
        self.lb_qu_path_image.setStyleSheet(u"QLabel {\n"
"    background-color: #dfd7cf;\n"
"}")
        self.lb_qu_path_image.setFrameShape(QFrame.WinPanel)
        self.lb_qu_path_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.lb_qu_path_image, 3, 1, 1, 1)

        self.label_36 = QLabel(self.groupBox_3)
        self.label_36.setObjectName(u"label_36")
        self.label_36.setMinimumSize(QSize(200, 30))
        self.label_36.setMaximumSize(QSize(200, 30))
        self.label_36.setFont(font)
        self.label_36.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_36.setFrameShape(QFrame.NoFrame)
        self.label_36.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_36, 1, 2, 1, 1)

        self.lb_cam_image = QLabel(self.groupBox_3)
        self.lb_cam_image.setObjectName(u"lb_cam_image")
        self.lb_cam_image.setMinimumSize(QSize(200, 200))
        self.lb_cam_image.setMaximumSize(QSize(200, 200))
        self.lb_cam_image.setStyleSheet(u"QLabel {\n"
"    background-color: #dfd7cf;\n"
"}")
        self.lb_cam_image.setFrameShape(QFrame.WinPanel)
        self.lb_cam_image.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.lb_cam_image, 3, 2, 1, 1)

        self.label_31 = QLabel(self.groupBox_3)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setMinimumSize(QSize(200, 30))
        self.label_31.setMaximumSize(QSize(200, 30))
        self.label_31.setFont(font)
        self.label_31.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_31.setFrameShape(QFrame.NoFrame)
        self.label_31.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_31, 1, 0, 1, 1)


        self.gridLayout_6.addWidget(self.groupBox_3, 1, 0, 1, 1)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(850, 575))
        self.groupBox.setMaximumSize(QSize(850, 575))
        self.groupBox.setStyleSheet(u"QGroupBox {\n"
"    background-color: rgba(42, 56, 82, 180);\n"
"    border-radius: 12px;\n"
"    border: none;\n"
"}")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.le_ROIs_focused = QLineEdit(self.groupBox)
        self.le_ROIs_focused.setObjectName(u"le_ROIs_focused")
        self.le_ROIs_focused.setMinimumSize(QSize(30, 30))
        self.le_ROIs_focused.setMaximumSize(QSize(30, 30))
        self.le_ROIs_focused.setFont(font)
        self.le_ROIs_focused.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.le_ROIs_focused, 4, 1, 1, 1)

        self.le_areas_focused = QLineEdit(self.groupBox)
        self.le_areas_focused.setObjectName(u"le_areas_focused")
        self.le_areas_focused.setMinimumSize(QSize(30, 30))
        self.le_areas_focused.setMaximumSize(QSize(30, 30))
        self.le_areas_focused.setFont(font)
        self.le_areas_focused.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.le_areas_focused, 3, 1, 1, 1)

        self.lb_cam_circle_image = QLabel(self.groupBox)
        self.lb_cam_circle_image.setObjectName(u"lb_cam_circle_image")
        self.lb_cam_circle_image.setMinimumSize(QSize(400, 400))
        self.lb_cam_circle_image.setMaximumSize(QSize(400, 400))
        self.lb_cam_circle_image.setStyleSheet(u"QLabel {\n"
"    background-color: #dfd7cf;\n"
"}")
        self.lb_cam_circle_image.setFrameShape(QFrame.WinPanel)
        self.lb_cam_circle_image.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.lb_cam_circle_image, 2, 0, 1, 4)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setMinimumSize(QSize(120, 30))
        self.label_3.setMaximumSize(QSize(130, 30))
        font12 = QFont()
        font12.setPointSize(6.5)
        font12.setBold(True)
        self.label_3.setFont(font12)
        self.label_3.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.label_23 = QLabel(self.groupBox)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setMinimumSize(QSize(400, 30))
        self.label_23.setMaximumSize(QSize(16777215, 30))
        self.label_23.setFont(font)
        self.label_23.setStyleSheet(u"QLabel {\n"
"    background-color: #309090;\n"
"    color: #F5F5F5;\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.label_23.setFrameShape(QFrame.WinPanel)
        self.label_23.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_23, 0, 0, 1, 7)

        self.label_16 = QLabel(self.groupBox)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setMinimumSize(QSize(400, 30))
        self.label_16.setMaximumSize(QSize(400, 30))
        self.label_16.setFont(font)
        self.label_16.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_16.setFrameShape(QFrame.NoFrame)
        self.label_16.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_16, 1, 0, 1, 4)

        self.lb_selected_focus_image = QLabel(self.groupBox)
        self.lb_selected_focus_image.setObjectName(u"lb_selected_focus_image")
        self.lb_selected_focus_image.setMinimumSize(QSize(400, 400))
        self.lb_selected_focus_image.setMaximumSize(QSize(400, 400))
        self.lb_selected_focus_image.setStyleSheet(u"QLabel {\n"
"    background-color: #dfd7cf;\n"
"}")
        self.lb_selected_focus_image.setFrameShape(QFrame.WinPanel)
        self.lb_selected_focus_image.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.lb_selected_focus_image, 2, 4, 1, 3)

        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMinimumSize(QSize(160, 60))
        self.label_4.setMaximumSize(QSize(160, 60))
        font1 = QFont()
        font1.setPointSize(8)
        font1.setBold(True)
        self.label_4.setFont(font1)
        self.label_4.setStyleSheet(u"QLabel {\n"
"    background-color:  #465272;  /* Dark red */\n"
"    color: #F5F5F5;               /* Black text */\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")

        self.gridLayout.addWidget(self.label_4, 3, 2, 2, 1)

        self.le_focus_relevance = QLineEdit(self.groupBox)
        self.le_focus_relevance.setObjectName(u"le_focus_relevance")
        self.le_focus_relevance.setMinimumSize(QSize(60, 60))
        self.le_focus_relevance.setMaximumSize(QSize(60, 60))
        self.le_focus_relevance.setFont(font1)
        self.le_focus_relevance.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.le_focus_relevance, 3, 3, 2, 1)

        self.le_focus_toggle = QLineEdit(self.groupBox)
        self.le_focus_toggle.setObjectName(u"le_focus_toggle")
        self.le_focus_toggle.setMinimumSize(QSize(45, 30))
        self.le_focus_toggle.setMaximumSize(QSize(45, 30))
        self.le_focus_toggle.setFont(font)
        self.le_focus_toggle.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.le_focus_toggle, 3, 5, 1, 1)

        self.label_29 = QLabel(self.groupBox)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setMinimumSize(QSize(400, 30))
        self.label_29.setMaximumSize(QSize(400, 30))
        self.label_29.setFont(font)
        self.label_29.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_29.setFrameShape(QFrame.NoFrame)
        self.label_29.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_29, 1, 4, 1, 3)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(120, 30))
        self.label_2.setMaximumSize(QSize(130, 30))
        font13 = QFont()
        font13.setPointSize(6.5)
        font13.setBold(True)
        self.label_2.setFont(font13)
        self.label_2.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")

        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(135, 30))
        self.label.setMaximumSize(QSize(135, 30))
        self.label.setFont(font1)
        self.label.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")

        self.gridLayout.addWidget(self.label, 3, 6, 1, 1)

        self.s_focus_slider = QSlider(self.groupBox)
        self.s_focus_slider.setObjectName(u"s_focus_slider")
        self.s_focus_slider.setMinimumSize(QSize(400, 30))
        self.s_focus_slider.setMaximumSize(QSize(400, 30))
        self.s_focus_slider.setValue(80)
        self.s_focus_slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.s_focus_slider, 4, 4, 1, 3)

        self.lb_focus_slider = QLabel(self.groupBox)
        self.lb_focus_slider.setObjectName(u"lb_focus_slider")
        self.lb_focus_slider.setMinimumSize(QSize(200, 30))
        self.lb_focus_slider.setMaximumSize(QSize(200, 30))
        self.lb_focus_slider.setFont(font)
        self.lb_focus_slider.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.lb_focus_slider.setFrameShape(QFrame.NoFrame)
        self.lb_focus_slider.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.lb_focus_slider, 3, 4, 1, 1)


        self.gridLayout_6.addWidget(self.groupBox, 0, 1, 3, 1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setMinimumSize(QSize(650, 0))
        self.groupBox_2.setMaximumSize(QSize(650, 275))
        self.groupBox_2.setStyleSheet(u"QGroupBox {\n"
"    background-color: rgba(42, 56, 82, 180);\n"
"    border-radius: 12px;\n"
"    border: none;\n"
"}")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.p_bar_concordant = QProgressBar(self.groupBox_2)
        self.p_bar_concordant.setObjectName(u"p_bar_concordant")
        self.p_bar_concordant.setMinimumSize(QSize(150, 30))
        self.p_bar_concordant.setMaximumSize(QSize(16777215, 30))
        self.p_bar_concordant.setFont(font)
        self.p_bar_concordant.setValue(0)

        self.gridLayout_2.addWidget(self.p_bar_concordant, 3, 2, 1, 1)

        self.p_bar_discordant = QProgressBar(self.groupBox_2)
        self.p_bar_discordant.setObjectName(u"p_bar_discordant")
        self.p_bar_discordant.setMinimumSize(QSize(150, 30))
        self.p_bar_discordant.setMaximumSize(QSize(16777215, 30))
        self.p_bar_discordant.setFont(font)
        self.p_bar_discordant.setValue(0)

        self.gridLayout_2.addWidget(self.p_bar_discordant, 4, 2, 1, 1)

        self.label_21 = QLabel(self.groupBox_2)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setMinimumSize(QSize(250, 30))
        self.label_21.setMaximumSize(QSize(250, 30))
        self.label_21.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_21.setFrameShape(QFrame.NoFrame)

        self.gridLayout_2.addWidget(self.label_21, 1, 0, 1, 2)

        self.label_19 = QLabel(self.groupBox_2)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setMinimumSize(QSize(325, 30))
        self.label_19.setMaximumSize(QSize(325, 30))
        self.label_19.setFont(font)
        self.label_19.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_19.setFrameShape(QFrame.NoFrame)
        self.label_19.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_19, 2, 0, 1, 2)

        self.label_22 = QLabel(self.groupBox_2)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setMinimumSize(QSize(325, 30))
        self.label_22.setMaximumSize(QSize(325, 30))
        self.label_22.setFont(font)
        self.label_22.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_22.setFrameShape(QFrame.NoFrame)
        self.label_22.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_22, 3, 0, 1, 2)

        self.label_20 = QLabel(self.groupBox_2)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setMinimumSize(QSize(400, 30))
        self.label_20.setMaximumSize(QSize(16777215, 30))
        self.label_20.setFont(font)
        self.label_20.setStyleSheet(u"QLabel {\n"
"    background-color: #309090;\n"
"    color: #F5F5F5;\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.label_20.setFrameShape(QFrame.WinPanel)
        self.label_20.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_20, 0, 0, 1, 3)

        self.label_18 = QLabel(self.groupBox_2)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setMinimumSize(QSize(150, 30))
        self.label_18.setMaximumSize(QSize(16777215, 30))
        self.label_18.setFont(font)
        self.label_18.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_18.setFrameShape(QFrame.NoFrame)
        self.label_18.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_18, 1, 2, 1, 1)

        self.p_bar_control = QProgressBar(self.groupBox_2)
        self.p_bar_control.setObjectName(u"p_bar_control")
        self.p_bar_control.setMinimumSize(QSize(150, 30))
        self.p_bar_control.setMaximumSize(QSize(16777215, 30))
        self.p_bar_control.setFont(font)
        self.p_bar_control.setValue(0)

        self.gridLayout_2.addWidget(self.p_bar_control, 2, 2, 1, 1)

        self.label_27 = QLabel(self.groupBox_2)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setMinimumSize(QSize(275, 60))
        self.label_27.setMaximumSize(QSize(275, 60))
        self.label_27.setFont(font)
        self.label_27.setStyleSheet(u"QLabel {\n"
"    color: #ffffff;\n"
"}")
        self.label_27.setFrameShape(QFrame.StyledPanel)
        self.label_27.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.label_27, 5, 1, 1, 1)

        self.lb_prediction = QLabel(self.groupBox_2)
        self.lb_prediction.setObjectName(u"lb_prediction")
        self.lb_prediction.setMinimumSize(QSize(250, 60))
        self.lb_prediction.setMaximumSize(QSize(16777215, 60))
        font2 = QFont()
        font2.setPointSize(16)
        font2.setBold(True)
        self.lb_prediction.setFont(font2)
        self.lb_prediction.setStyleSheet(u"QLabel {\n"
"    background-color: #8E9AAF;  /* Dark red */\n"
"    color: #F5F5F5;               /* Black text */\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.lb_prediction.setFrameShape(QFrame.StyledPanel)
        self.lb_prediction.setAlignment(Qt.AlignCenter)

        self.gridLayout_2.addWidget(self.lb_prediction, 5, 2, 1, 1)

        self.label_24 = QLabel(self.groupBox_2)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setMinimumSize(QSize(325, 30))
        self.label_24.setMaximumSize(QSize(325, 30))
        self.label_24.setFont(font)
        self.label_24.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_24.setFrameShape(QFrame.NoFrame)
        self.label_24.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_24, 4, 0, 1, 2)


        self.gridLayout_6.addWidget(self.groupBox_2, 2, 0, 2, 1)

        self.groupBox_4 = QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setMinimumSize(QSize(650, 125))
        self.groupBox_4.setMaximumSize(QSize(650, 125))
        self.groupBox_4.setStyleSheet(u"QGroupBox {\n"
"    background-color: rgba(42, 56, 82, 180);\n"
"    border-radius: 12px;\n"
"    border: none;\n"
"}")
        self.gridLayout_5 = QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.label_42 = QLabel(self.groupBox_4)
        self.label_42.setObjectName(u"label_42")
        self.label_42.setMinimumSize(QSize(200, 30))
        self.label_42.setMaximumSize(QSize(16777215, 30))
        self.label_42.setBaseSize(QSize(0, -1))
        self.label_42.setFont(font)
        self.label_42.setStyleSheet(u"QLabel {\n"
"    background-color: #309090;\n"
"    color: #F5F5F5;\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.label_42.setFrameShape(QFrame.WinPanel)
        self.label_42.setAlignment(Qt.AlignCenter)

        self.gridLayout_5.addWidget(self.label_42, 0, 0, 1, 3)

        self.label_44 = QLabel(self.groupBox_4)
        self.label_44.setObjectName(u"label_44")
        self.label_44.setMinimumSize(QSize(200, 30))
        self.label_44.setMaximumSize(QSize(200, 30))
        self.label_44.setFont(font)
        self.label_44.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_44.setFrameShape(QFrame.NoFrame)

        self.gridLayout_5.addWidget(self.label_44, 1, 0, 1, 1)

        self.label_46 = QLabel(self.groupBox_4)
        self.label_46.setObjectName(u"label_46")
        self.label_46.setMinimumSize(QSize(200, 30))
        self.label_46.setMaximumSize(QSize(200, 30))
        self.label_46.setFont(font)
        self.label_46.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_46.setFrameShape(QFrame.NoFrame)

        self.gridLayout_5.addWidget(self.label_46, 1, 1, 1, 1)

        self.label_47 = QLabel(self.groupBox_4)
        self.label_47.setObjectName(u"label_47")
        self.label_47.setMinimumSize(QSize(200, 30))
        self.label_47.setMaximumSize(QSize(200, 30))
        self.label_47.setFont(font)
        self.label_47.setStyleSheet(u"QLabel {\n"
"    color: #F5F5F5;\n"
"}")
        self.label_47.setFrameShape(QFrame.NoFrame)

        self.gridLayout_5.addWidget(self.label_47, 1, 2, 1, 1)

        self.lb_case_id = QLineEdit(self.groupBox_4)
        self.lb_case_id.setObjectName(u"lb_case_id")
        self.lb_case_id.setMinimumSize(QSize(200, 30))
        self.lb_case_id.setMaximumSize(QSize(200, 30))

        self.gridLayout_5.addWidget(self.lb_case_id, 2, 0, 1, 1)

        self.lb_image_no = QLineEdit(self.groupBox_4)
        self.lb_image_no.setObjectName(u"lb_image_no")
        self.lb_image_no.setMinimumSize(QSize(200, 30))
        self.lb_image_no.setMaximumSize(QSize(200, 30))

        self.gridLayout_5.addWidget(self.lb_image_no, 2, 1, 1, 1)

        self.lb_region = QLineEdit(self.groupBox_4)
        self.lb_region.setObjectName(u"lb_region")
        self.lb_region.setMinimumSize(QSize(200, 30))
        self.lb_region.setMaximumSize(QSize(200, 30))

        self.gridLayout_5.addWidget(self.lb_region, 2, 2, 1, 1)


        self.gridLayout_6.addWidget(self.groupBox_4, 0, 0, 1, 1)

        self.groupBox_5 = QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setMinimumSize(QSize(0, 0))
        self.groupBox_5.setMaximumSize(QSize(16777215, 16777215))
        self.groupBox_5.setStyleSheet(u"QGroupBox {\n"
"    background-color: rgba(42, 56, 82, 180);\n"
"    border-radius: 12px;\n"
"    color: #F5F5F5\n"
"    border: none;\n"
"}")
        self.gridLayout_3 = QGridLayout(self.groupBox_5)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.label_48 = QLabel(self.groupBox_5)
        self.label_48.setObjectName(u"label_48")
        self.label_48.setMinimumSize(QSize(200, 30))
        self.label_48.setMaximumSize(QSize(16777215, 30))
        self.label_48.setFont(font)
        self.label_48.setStyleSheet(u"QLabel {\n"
"    background-color: #309090;\n"
"    color: #F5F5F5;\n"
"    padding: 4px;\n"
"    border-radius: 4px;\n"
"}")
        self.label_48.setFrameShape(QFrame.WinPanel)
        self.label_48.setAlignment(Qt.AlignCenter)

        self.gridLayout_3.addWidget(self.label_48, 0, 0, 1, 1)

        self.te_notes = QTextEdit(self.groupBox_5)
        self.te_notes.setObjectName(u"te_notes")

        self.gridLayout_3.addWidget(self.te_notes, 1, 0, 1, 1)


        self.gridLayout_6.addWidget(self.groupBox_5, 3, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1533, 37))
        self.menubar.setStyleSheet(u"QMenuBar {\n"
"    background-color: #182029;\n"
"    color: #F5F5F5;\n"
"    font-size: 12pt;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QMenuBar::item {\n"
"    background-color: transparent;\n"
"    color: #F5F5F5;\n"
"    padding: 4px 10px;\n"
"    font-size: 12pt;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QMenuBar::item:selected {\n"
"    background-color: #2b3952; /* Slightly darker on hover */\n"
"    color: #F5F5F5;\n"
"    font-size: 12pt;\n"
"    font-weight: bold;\n"
"}\n"
"")
        self.menuControls = QMenu(self.menubar)
        self.menuControls.setObjectName(u"menuControls")
        MainWindow.setMenuBar(self.menubar)

        self.menubar.addAction(self.menuControls.menuAction())
        self.menuControls.addAction(self.a_browse_image)
        self.menuControls.addAction(self.a_export_results)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.a_browse_image.setText(QCoreApplication.translate("MainWindow", u"Browse Image", None))
        self.a_run_analysis.setText(QCoreApplication.translate("MainWindow", u"Run Analysis", None))
        self.a_export_results.setText(QCoreApplication.translate("MainWindow", u"Export Results", None))
        self.a_clear.setText(QCoreApplication.translate("MainWindow", u"clear", None))
        self.groupBox_3.setTitle("")
        self.label_33.setText(QCoreApplication.translate("MainWindow", u"QuPath detected ROIs in Image", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"Model Focus Images", None))
        self.lb_input_image.setText(QCoreApplication.translate("MainWindow", u"Drag or Browse \n an Image To Get Started", None))
        self.lb_qu_path_image.setText(QCoreApplication.translate("MainWindow", u"Waiting for Image", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"LayerCAM Heat-Map", None))
        self.lb_cam_image.setText(QCoreApplication.translate("MainWindow", u"Waiting for Image", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Original Input Image", None))
        self.groupBox.setTitle("")
        self.le_ROIs_focused.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.le_areas_focused.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.lb_cam_circle_image.setText(QCoreApplication.translate("MainWindow", u"Waiting for Image", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"No. Areas Overlapping QuPath ROIs", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Attention Analysis Panel", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"LayerCAM Attention on Pathology Annotations", None))
        self.lb_selected_focus_image.setText(QCoreApplication.translate("MainWindow", u"Waiting for Image", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Annotation Agreement Score (%)", None))
        self.le_focus_relevance.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.le_focus_toggle.setText(QCoreApplication.translate("MainWindow", u"80", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"LayerCAM Attention on Raw Tissue", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"No. LayerCam Highlighted Regions", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"%", None))
        self.lb_focus_slider.setText(QCoreApplication.translate("MainWindow", u"LayerCAM Attention Threshold (%)", None))
        self.groupBox_2.setTitle("")
        self.label_21.setText("")
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Control (Healthy) :", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Concordant (ALS with Cognitive Imparment) :", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Model Analysis", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Model Prediction Confidence ", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Model Prediction ->", None))
        self.lb_prediction.setText(QCoreApplication.translate("MainWindow", u"Waiting...", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"Discordant(ALS without Cognitive Imparment) :", None))
        self.groupBox_4.setTitle("")
        self.label_42.setText(QCoreApplication.translate("MainWindow", u"Image Information", None))
        self.label_44.setText(QCoreApplication.translate("MainWindow", u"Case ID:", None))
        self.label_46.setText(QCoreApplication.translate("MainWindow", u"Image ID:", None))
        self.label_47.setText(QCoreApplication.translate("MainWindow", u"Region:", None))
        self.groupBox_5.setTitle("")
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"Case Information", None))
        self.menuControls.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
    # retranslateUi

