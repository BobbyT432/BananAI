import sys
import tensorflow as tf
import numpy as np
import os
import cv2
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from tensorflow.keras.models import Sequential # consider functional for multiple outputs
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(__file__))
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'mainwindow.ui'), resource_suffix='')

qtCreatorFile = "mainwindow.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class mainwindow(QMainWindow):
    fnameImg = None # class level var
    model = load_model(os.path.join('models','binarybanana.h5'))
    def __init__(self):
        super(mainwindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.processImg.hide()
        self.ui.picLabel2.hide()
        test = self.ui.pushButton1.clicked.connect(self.browseImage)
        print(test)
        self.ui.processImg.clicked.connect(self.process_img)
        self.ui.picLabel.setAlignment(Qt.AlignCenter)
    
    def process_img(self):
        if self.fnameImg is not None:
            model = self.model
            img = cv2.imread(self.fnameImg)
            resize = tf.image.resize(img, (256, 256))
            #model = load_model(os.path.join('models','binarybanana.h5'))
            yhat = model.predict(np.expand_dims(resize/255,0))
            self.ui.processImg.hide()
            self.ui.picLabel2.show()
            if yhat > 0.5:
                imagePath = "img/unripe_logo.png"
                pixmap = QPixmap(imagePath)
                height = self.ui.picLabel2.frameGeometry().height()
                width = self.ui.picLabel2.frameGeometry().width()
                pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
                self.ui.picLabel2.setPixmap(QPixmap(pixmap))
            else:
                imagePath = "img/ripe_logo.png"
                pixmap = QPixmap(imagePath)
                height = self.ui.picLabel2.frameGeometry().height()
                width = self.ui.picLabel2.frameGeometry().width()
                pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
                self.ui.picLabel2.setPixmap(QPixmap(pixmap))

    def browseImage(self):
        self.ui.picLabel2.hide()
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'c\\', 'Image files (*.jpg *.png)')
        self.fnameImg = fname[0]
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)

        #self.ui.picLabel.resize(pixmap.width()/widthAdj, pixmap.height()/heightAdj)
        # instead lets try to resize the image
        height = self.ui.picLabel.frameGeometry().height()
        width = self.ui.picLabel.frameGeometry().width()
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio) # just using the height and width of the actual thingy
        self.ui.picLabel.setPixmap(QPixmap(pixmap))
        self.ui.picLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.picLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.processImg.show()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainwindow()
    window.show()
    sys.exit(app.exec_())