from PyQt5.QtWidgets import QApplication,QMessageBox,QDesktopWidget,QMainWindow,QPushButton,QLabel, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from scipy import ndimage
import re
import sys
from PIL import Image, ImageEnhance
# from skimage.util import random_noise

from random import randint


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # No external window yet for load image function.
        self.w = None
        # No external window yet for inputs of crop image function.
        self.inputWindow = None
        # No external window yet for inputs of rotate image function.
        self.inputWindowOfRotation = None
        # chance color balance window
        self.ccbWindow = None
        # adjust brightness window
        self.adjBrgWindow = None
        # adjust saturation window
        self.adjSatWindow = None
        # flip image window
        self.flipImage = None
        # adjust contrast windpw
        self.adjConWindow = None

        self.warning = None
        xtop = 10
        yleft = 10
        margin = 10
        buttonWidth = 150
        buttonHeight = 50

        sizeObject = QDesktopWidget().screenGeometry()

        # access screen dimensions for image - screen accordance
        self.width = int(sizeObject.getRect()[2])
        self.height = int(sizeObject.getRect()[3])

        self.imgWidth = 0
        self.imgHeight = 0

        # set up message area widget
        # This for guiding user when deal with an error
        self.message = QLabel(self)
        self.message.setFont(QFont("Arial", 12))
        self.message.setGeometry(920, 60, 700, 200)
        self.message.setStyleSheet("color: red")

        # File text area widget
        hFile = QLabel(self)
        hFile.setText("File")
        hFile.setFont(QFont("Arial", 20))
        hFile.move(xtop+int(buttonWidth/4), yleft)

        # laod button widget
        # create new button
        loadButton = QPushButton(self)
        # set text of button
        loadButton.setText("Load Image")
        # set button coordinates and its width, height
        loadButton.setGeometry(xtop, yleft+buttonHeight, buttonWidth, buttonHeight)
        # runs function (show_new_window) when clicked button
        loadButton.clicked.connect(self.show_new_window)

        # save button widget
        saveButton = QPushButton(self)
        saveButton.setText("Save Image")
        saveButton.setGeometry(xtop, yleft+2*buttonHeight, buttonWidth, buttonHeight)
        saveButton.clicked.connect(self.save)

        # ------ EDIT AREA
        # Edit text area widget
        hEdit = QLabel(self)
        hEdit.setText("Edit")
        hEdit.setFont(QFont("Arial", 20))
        hEdit.move(buttonWidth+yleft+int(buttonWidth/4),xtop)

        # blur button widget
        blurButton = QPushButton(self)
        blurButton.setText("Blur Image")
        blurButton.setGeometry(xtop+buttonWidth, yleft+buttonHeight, buttonWidth, buttonHeight)
        blurButton.clicked.connect(self.blur)

        # deblur button widget
        deblurButton = QPushButton(self)
        deblurButton.setText("Deblur Image")
        deblurButton.setGeometry(xtop+buttonWidth, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        deblurButton.clicked.connect(self.deblur)

        # reverse color button widget
        reverseColorButton = QPushButton(self)
        reverseColorButton.setText("Reverse Color")
        reverseColorButton.setGeometry(xtop+buttonWidth, yleft+buttonHeight*3, buttonWidth, buttonHeight)
        reverseColorButton.clicked.connect(self.reverseColor)

        # grayscale button widget
        grayScaleButton = QPushButton(self)
        grayScaleButton.setText("Grayscale Image")
        grayScaleButton.setGeometry(xtop+buttonWidth*2, yleft+buttonHeight, buttonWidth, buttonHeight)
        grayScaleButton.clicked.connect(self.grayscale)

        # crop button widget
        cropButton = QPushButton(self)
        cropButton.setText("Crop Image")
        cropButton.setGeometry(xtop+buttonWidth*2, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        cropButton.clicked.connect(self.crop)

        # flip button widget
        flipButton = QPushButton(self)
        flipButton.setText("Flip Image")
        flipButton.setGeometry(xtop+buttonWidth*2, yleft+buttonHeight*3, buttonWidth, buttonHeight)
        flipButton.clicked.connect(self.flip)

        # mirror button widget
        mirrorButton = QPushButton(self)
        mirrorButton.setText("Mirror Image")
        mirrorButton.setGeometry(xtop+buttonWidth*3, yleft+buttonHeight, buttonWidth, buttonHeight)
        mirrorButton.clicked.connect(self.mirror)

        # rotate button widget
        rotateButton = QPushButton(self)
        rotateButton.setText("Rotate Image")
        rotateButton.setGeometry(xtop+buttonWidth*3, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        rotateButton.clicked.connect(self.rotate)

        # ccb: change color balance
        ccbButton = QPushButton(self)
        ccbButton.setText("Change Color Balance")
        ccbButton.setGeometry(xtop+buttonWidth*3, yleft+buttonHeight*3, buttonWidth, buttonHeight)
        ccbButton.clicked.connect(self.ccb)

        # adjBrg: adjust brightness
        adjBrgButton = QPushButton(self)
        adjBrgButton.setText("Adjust Brightness")
        adjBrgButton.setGeometry(xtop+buttonWidth*4, yleft+buttonHeight, buttonWidth, buttonHeight)
        adjBrgButton.clicked.connect(self.adjBrg)

        # adjSat: adjust Saturation
        adjBrgButton = QPushButton(self)
        adjBrgButton.setText("Adjust Saturation")
        adjBrgButton.setGeometry(xtop+buttonWidth*4, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        adjBrgButton.clicked.connect(self.adjSat)

        # Detect Edges of Image Button widget
        detectEdgesButton = QPushButton(self)
        detectEdgesButton.setText("Detect Edges")
        detectEdgesButton.setGeometry(xtop+buttonWidth*4, yleft+buttonHeight*3, buttonWidth, buttonHeight)
        detectEdgesButton.clicked.connect(self.detectEdges)

        # Add Noise to Image Button widget
        addNoiseButton = QPushButton(self)
        addNoiseButton.setText("Add Noise")
        addNoiseButton.setGeometry(xtop+buttonWidth*5, yleft+buttonHeight, buttonWidth, buttonHeight)
        addNoiseButton.clicked.connect(self.addNoise)

        # adjCon: adjustContrast
        adjConButton = QPushButton(self)
        adjConButton.setText("Adjust Contrast")
        adjConButton.setGeometry(xtop+buttonWidth*5, yleft+buttonHeight*2, buttonWidth, buttonHeight)
        adjConButton.clicked.connect(self.adjCon)


        # Loaded Image widget
        self.loadedImage = QLabel(self)
        # Scale image for screen accordance
        self.loadedImage.setScaledContents(True)
        self.loadedImage.setFixedHeight(int(self.height-250))
        self.loadedImage.setFixedWidth(int(self.width/2))
        self.loadedImagePath = ""
        self.loadedImage.move( 0,250)

        # manipulated Image widget
        self.manipulatedImage = QLabel(self)
        # Scale image for screen accordance
        self.manipulatedImage.setScaledContents(True)
        self.manipulatedImage.setFixedHeight(int(self.height - 250))
        self.loadedImage.setFixedWidth(int(self.width / 2))
        self.manipulatedImage.move(
            int(self.width/2),250)

        # set coordinate and sizes of main screen of application
        self.setGeometry(0, 0, self.width, self.height)
        self.setWindowTitle("Course Project for BBM413 and BBM415")

    def show_new_window(self, checked):

        if self.w is None:
            self.w = QFileDialog.Options()
            # get filename of image
            fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                      "All Files (*.jpg *.png *.jpeg)", options=self.w)

            # Original Image Widget
            pixmap = QPixmap(fileName)
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
            self.loadedImage.setPixmap(pixmap2)
            self.loadedImage.adjustSize()
            self.loadedImagePath = fileName

        self.w = None


    def save(self):
        try:
            # if path is empty, raise FileNotFoundError
            if(len(self.loadedImagePath) == 0):
                raise FileNotFoundError
            self.manipulatedImage.pixmap().save("savedImage.jpg","JPG")
            self.message.setText("")

        # display error message
        except FileNotFoundError:
            self.message.setText("You have to create manipulated image to save it!")
        except Exception as E:
            self.message.setText(str(E))

    def blur(self):

        # QCoreApplication.exit(0)

        try:
            # access loaded Image
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError
            # blur image
            blurImg = cv2.blur(image, (9, 9))

            # save blurred image temporarily
            cv2.imwrite("temp.jpg", blurImg)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            # set message text to empty, when process s successfull
            self.message.setText("")
        except FileNotFoundError:
            self.message.setText("You have to load an image before blur!")
        except Exception as E:
            self.message.setText(str(E))

    def deblur(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(image, -1, sharpen_kernel)

            cv2.imwrite("temp.jpg", sharpen)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()

            self.message.setText("")

        except FileNotFoundError:
            self.message.setText("You have to load an image before deblur!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)
    def reverseColor(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            # if image is None, raise FileNotFoundError
            if(image is None):
                raise FileNotFoundError

            # reverse color
            image = (255 - image)
            cv2.imwrite("temp.jpg", image)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        #   display relevent error message in ui
        except FileNotFoundError:
            self.message.setText("You have to load an image before reverse color!")
        except Exception as E:
            self.message.setText(str(E))

    def grayscale(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("temp.jpg", gray_image)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("You have to load an image before grayscale!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def crop(self):
        if self.inputWindow is None:
            try:
                # self.inputWindow = QWidget()
                if(len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                start, okPressed = QInputDialog. \
                    getText(self, "Get coordinates", "ENTER STARTING POINT (TOP LEFT):\n in format x,y\n for example 120,85", QLineEdit.Normal, "",)
                end, okPressed = QInputDialog. \
                    getText(self, "Get coordinates", "ENTER ENDING POINT (BOTTOM RIGHT):\n in format x,y\n for example 120,85",
                            QLineEdit.Normal, "")

                image = cv2.imread(self.loadedImagePath)
                if(image is None):
                    raise FileNotFoundError
                if(len(start.split(",")) != 2 or len(end.split(",")) != 2 ):
                    raise Exception

                image = np.array(image)

                # coordinates of starting point(top-left)
                x1 = int(start.split(",")[0])
                y1 = int(start.split(",")[1])

                # coordinates of end point(right-bottom)
                x2 = int(end.split(",")[0])
                y2 = int(end.split(",")[1])

                # crop image with inputs
                croppedImage = image[x1:x2, y1:y2]

                cv2.imwrite("temp.jpg", croppedImage)
                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before crop!")
            except Exception as E:
                self.message.setText("Invalid Input, \nit can be a good idea to review pixel size of original image by giving inputs ")
                print(E)

    def flip(self):
        if(self.flipImage is None):
            try:
                flipValue, okPressed = QInputDialog.getText(self, "Rotation", "Enter value for your options\n"
                                                                              "0: Vertical Flip\n"
                                                                              "1: Horizontal Flip", QLineEdit.Normal, "",)
                image = cv2.imread(self.loadedImagePath)
                if(image is None):
                    raise FileNotFoundError
                elif(int(flipValue)>1):
                    raise Exception

                # second argument of cv2.flip is horizontal or vertical
                # 0 for vertical flip
                # 1 for horizontal flip
                flippedImage = cv2.flip(image,int(flipValue))

                cv2.imwrite("temp.jpg", flippedImage)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
                self.message.setText("")
            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before flip!")
            except Exception as E:
                self.message.setText("Invalid Input")
                print(E)

    def mirror(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            mirroredImage = cv2.flip(image,1)
            cv2.imwrite("temp.jpg", mirroredImage)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")
        # display relevant error
        except FileNotFoundError:
            self.message.setText("You have to load an image before mirror!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def rotate(self):
        if self.inputWindowOfRotation is None:
            try:
                if(len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                rotationDegree, okPressed = QInputDialog.getText(self, "Rotation", "Enter Rotation Degree", QLineEdit.Normal, "",)

                image = cv2.imread(self.loadedImagePath)
                rotatedImage = ndimage.rotate(image, int(rotationDegree))

                cv2.imwrite("temp.jpg", rotatedImage)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
                self.message.setText("")

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before rotation!")
            except Exception as E:
                self.message.setText(str(E))
                print(E)

    def ccb(self):
        if self.warning is None:
            self.message.setText("This process can take a few seconds!\n Please wait")

        if self.inputWindow is None:
            try:
                if(len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                newRGB, okPressed = QInputDialog. \
                    getText(self, "Change Color Balance", ("enter your increase or decrease value in format r,g,b\n For example:112,144,96 or 112,-50,-32"), QLineEdit.Normal, "",)
                image = cv2.imread(self.loadedImagePath)

                ccbImage = image
                # split new rgb values
                newR,newG,newB = newRGB.split(",")[0] , newRGB.split(",")[1] , newRGB.split(",")[2]

                # add new rgb value pixel by pixel
                for row in ccbImage:
                    for pixel in row:
                        if ((int(newR) + pixel[2]) / 255) >= 1:
                            pixel[2] = 255
                        elif((int(newR) + pixel[2]) < 0):
                            pixel[2] = 0
                        else:
                            pixel[2] = pixel[2]+int(newR)

                        if ((int(newG) + pixel[1]) / 255) >= 1:
                            pixel[1] = 255
                        elif ((int(newG) + pixel[1]) < 0):
                            pixel[1] = 0

                        else:
                            pixel[1] = pixel[1]+int(newG)

                        if ((int(newB) + pixel[0]) / 255) >= 1:
                            pixel[0] = 255
                        elif ((int(newB) + pixel[0]) < 0):
                            pixel[0] = 0
                        else:
                            pixel[0] = pixel[0]+int(newB)

                cv2.imwrite("temp.jpg", ccbImage)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
                self.message.setText("")

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before change color balance!")
            except Exception as E:
                self.message.setText("Invalid Input")
                print(E)
    def adjBrg(self):
        if self.adjBrgWindow is None:
            try:
                if (len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                value, okPressed = QInputDialog. \
                    getText(self, "Adjust Brightness",
                            "Enter negative or positive brightness value:\n(default value is 0)", QLineEdit.Normal, "", )

                image = cv2.imread(self.loadedImagePath)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                h, s, v = cv2.split(hsv)

                # find value without its sign mark
                val = int(re.findall('\d+', value)[0])

                if(int(value)>0):
                    v = cv2.add(v,int(val))
                else:
                    v = cv2.subtract(v,int(val))

                v[v > 255] = 255
                v[v < 0] = 0
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


                cv2.imwrite("temp.jpg", img)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))
                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before adjust brightness!")
            except Exception as E:
                self.message.setText("invalid input")
                # self.message.setText(str(E))
                print(E)
    def adjSat(self):
        if self.adjSatWindow is None:
            try:
                if (len(self.loadedImagePath) == 0):
                    raise FileNotFoundError
                satVal, okPressed = QInputDialog. \
                    getText(self, "Adjust Saturation",
                            "Enter saturation value: (Make sure you put a point instead of a comma if you use float numbers!)\n"
                            "(defalut value is 1)", QLineEdit.Normal, "", )

                image = Image.open(self.loadedImagePath)

                converter = ImageEnhance.Color(image)
                img2 = converter.enhance(float(satVal))


                last = np.array(img2)
                last  = cv2.cvtColor(last, cv2.COLOR_RGB2BGR)
                cv2.imwrite("temp.jpg", last)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
                self.message.setText("")

            # display relevant error message
            except FileNotFoundError:
                self.message.setText("You have to load an image before adjust saturation!")
            except Exception as E:
                self.message.setText(str(E))
                print(E)

    def detectEdges(self):

        try:
            img = cv2.imread(self.loadedImagePath)
            if(img is None):
                raise FileNotFoundError
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection


            edges = np.array(edges)
            cv2.imwrite("temp.jpg", edges)

            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("You have to load an image before detect edges!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def addNoise(self):
        try:
            image = cv2.imread(self.loadedImagePath)
            if(image is None):
                raise FileNotFoundError

            gauss = np.random.normal(0, 1, image.size)
            gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
            mode = "speckle"

            if mode == "gaussian":
                img_gauss = cv2.add(image, gauss)

                cv2.imwrite("./temp.jpg", img_gauss)
                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()

            elif mode == "speckle":
                noise = image + image * gauss

            cv2.imwrite("./temp.jpg", noise)
            pixmap = QPixmap("./temp.jpg")
            pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

            self.manipulatedImage.setPixmap(pixmap2)
            self.manipulatedImage.adjustSize()
            # cv2.imshow('ab', noise)
            # cv2.waitKey()
            # return noise
            self.message.setText("")

        # display relevant error message
        except FileNotFoundError:
            self.message.setText("You have to load an image before add noise!")
        except Exception as E:
            self.message.setText(str(E))
            print(E)

    def adjCon(self):
        if(self.adjConWindow is None):
            try:
                image = cv2.imread(self.loadedImagePath, 1)
                if(image is None):
                    raise FileNotFoundError

                conVal, okPressed = QInputDialog. \
                    getText(self, "Adjust Contrast",
                            "Enter contrast value: (Make sure you put a point instead of a comma if you use float numbers!)\n"
                            "(defalut value is 1)", QLineEdit.Normal, "", )

                clahe = cv2.createCLAHE(clipLimit=float(conVal), tileGridSize=(8, 8))
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
                l, a, b = cv2.split(lab)  # split on 3 different channels
                l2 = clahe.apply(l)  # apply CLAHE to the L-channel
                lab = cv2.merge((l2, a, b))  # merge channels
                img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

                cv2.imwrite("./temp.jpg", img2)

                pixmap = QPixmap("./temp.jpg")
                pixmap2 = pixmap.scaledToWidth(int(self.width / 2))

                self.manipulatedImage.setPixmap(pixmap2)
                self.manipulatedImage.adjustSize()
            except FileNotFoundError:
                self.message.setText("you have to load an image before adjust contrast")
            except Exception as E:
                self.message.setText(str(E))
app = QApplication(sys.argv)
main = MainWindow()
main.show()
app.exec()