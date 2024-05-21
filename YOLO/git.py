import cv2 as cv  # this is the computer vision header
import cv2 #also cv header
import time  # time header
import numpy as np  # numpy/matrix header
import os 
from matplotlib import pyplot as plt


def ReadPicture():
    img = cv.imread('DogZiller.jpg')
    # this reads the image. Put the image name in single quotes

    if img is None:
        print("Error: Unable to read the image file")
        return

    cv2.imshow('DogZiller', img)
    # this shows the image
    #this name "DogZiller" will be the title of the pop up window

    cv2.waitKey(0)
    # this sets the "wait time" infinite by setting it to 0


def ReadVideo():
    capture = cv.VideoCapture('YOLO\Dogzilla_Video.mp4')
    # This reads the video. Argument can be an integer (for webcam) or a video path.
    # you can also source the relative path as I did above 'YOLO\Dogzilla_Video.mp4'

    if not capture.isOpened():
        print("Error: Unable to open video file")
        return

    while True:
        # Need to read video in a while loop
        isTrue, frame = capture.read()
        # capture.read() reads the video frame by frame.
        # frame returns the current frame.
        # isTrue returns True if the frame was successfully read, False otherwise.

        if not isTrue:
            print("End of video or unable to read the frame")
            # If the frame was not successfully read, exit the loop
            break

        cv2.imshow('Dogzilla_Video', frame)
        # This will show the video. Can replace 'Dogzilla_Video' with '0' for webcam.
        #this name "DogZilla_Video" will be the title of the pop up window


        if cv2.waitKey(20) & 0xFF == ord('d'):
            # If the letter 'd' is pressed, end the video.
            break

    capture.release()
    cv2.destroyAllWindows()
    # Closes the video window

def ResizePicture_ParameterWay(frame, scale = 0.75):
    #this method is by using the function parameters unlike the one below
    #takes in the frame (picture) and scales it by 0.75
    #Im just using 0.75 as an example

    img = cv2.imread('DogZiller.jpg') #everything from here

    if img is None:
        print("Error: Unable to read the image file")
        return

    cv2.imshow('Title', img) # to here need to be OUTSIDE and BEFORE this resize function 

    width = int(frame.shape[1]* scale)
    height = int(frame.shape[1]*scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    cv2.waitKey(0)

def ResizePic_EasyWay():
    #this is the "better" way

    img = cv2.imread('DogZiller.jpg')
    
    Resized_Pic = cv2.resize(img, (200,100))
    # notice how its (width, height) but when you print it, its (height, width)
    
    print(Resized_Pic.shape)
    #prints (100 pixels height, 200 pixels width, Number of Color Changes)

    # Display the captured frame
    cv2.imshow('Resizded Pic', Resized_Pic)
    
    cv2.waitKey(0)


def CropPic():
    #this is how to crop a pic
    
    img = cv2.imread('DogZiller.jpg')
    
    Cropped_Pic = img[320:640, 420:840]
    #img[width intervals : height intervals]
    # Meaning this Crop Pic will be cropped between 320-640 pixel W and 420-840 pixels H
    
    print(Cropped_Pic.shape)
    #prints (pixel height,pixels width, Number of Color Changes)
    
    cv2.imshow('Original Pic', img)
    #show the origional pic

    cv2.imshow('Cropped Pic', Cropped_Pic)
    #show the new cropped pic
    
    cv2.waitKey(0)
    
def ConvertColor():

    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #color convertor, in this example we're going from Blue Green Red (BGR) to RGB
    #you should notice that all original blues are now red, all old reds are now blue
    # and green stayed the same 
    #this robot pic is a terrible example so use your own

    cv2.imshow('Origional Pic', img)
    #show the old pic before conversion 

    cv2.imshow('RGB Conversion', img_rbg)
    #Show the new pic with RGB conversion


    cv2.waitKey(0)
    

def GreyScale():

    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #turns the image grey
    #as you can see by the auto fill, there are MANY COLOR_XYZ2XYZ color conversions
    
    #here is a link to all color conversions via OpenCV - https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Grey Scale Conversion', img_greyscaled)
    #Show the new pic with RGB conversion


    cv2.waitKey(0)

def General_Blurring():


    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    k_size = 7
    #the larget this number, the larger the blur
    
    #more information here 
    #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    img_blur = cv2.blur(img,(k_size ,k_size ))

    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Blurred Image', img_blur)
    #Show the new pic with  conversion


    cv2.waitKey(0)


def Gaussian_Blurring():


    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    k_size = 7
    #the larget this number, the larger the blur
    
    #more information here 
    #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    img_blur = cv2.blur(img,(k_size ,k_size ))
    
    img_Guas = cv2.GaussianBlur(img, (k_size, k_size), 3)
    #The standard deviation along the x and y directions of the Gaussian kernel. 
    # This controls the spread of the Gaussian distribution used for blurring. 
    # A higher standard deviation results in a blurrier image.
    

    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Guassian Blurred Image', img_Guas)
    #Show the new pic with  conversion


    cv2.waitKey(0)


def Median_Blurring():


    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    k_size = 7
    #the larget this number, the larger the blur
    
    #more information here 
    #https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    img_blur = cv2.blur(img,(k_size ,k_size ))
    
    img_Guas = cv2.GaussianBlur(img, (k_size, k_size), 3)
    #The standard deviation along the x and y directions of the Gaussian kernel. 
    # This controls the spread of the Gaussian distribution used for blurring. 
    # A higher standard deviation results in a blurrier image.
    
    img_median_blur = cv2.medianBlur(img, k_size)
    

    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Median Blurred Image', img_median_blur)
    #Show the new pic with  conversion


    cv2.waitKey(0)


def Simple_Thresholding():
    
    #supporting documents 
    
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(img_greyscaled, 80, 255, cv2.THRESH_BINARY)
    #all values below 80 will go to 0 an anything above 80 will go to 255
    
    #this example is using threshold binary but there are many types from the link above
    
    thresh = cv2.blur(thresh,(10,10))                                   #can delete this
    #10 and 10 are the k sizes used (just an example)
    
    ret, thresh = cv2.threshold(thresh, 80, 255, cv2.THRESH_BINARY)     #can delete this
    
    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Threshold Conversion', thresh)
    #Show the new pic with RGB conversion

    cv2.waitKey(0)
    
def Adaptive_Thresholding():
    
    #adaptive thresholding is good when you have shadows or lots of black and white in an image/video
    
    #supporting documents 
    
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(img_greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,30)
    
    #this is using the adaptive GAUSSIAN blue, there are others
    
    #the 20 and 30 are the thresholds for the regions
    
    #s the threshold is 20 or below, it will apply a gaussian and 
    # if the threshhold is abovfe 31, then it will apply a binary threshold

    cv2.imshow('Original Pic', img)
    #show the old pic before conversion 

    cv2.imshow('Adapt Threshold Conversion', thresh)
    #Show the new pic with RGB conversion

    cv2.waitKey(0)
    
def Edge_Detection():
    
    #Canny Edge Detection from Open CV
    
    # https://docs.opencv.org/4.x/d7/de1/tutorial_js_canny.html
    
    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_edge = cv2.Canny(img, 100,200)
    
    '''
    The first threshold (100) is the lower threshold. It is used to identify weak edges in the image. 
    Any edge with a gradient magnitude higher than this threshold is considered a strong edge.

    The second threshold (200) is the higher threshold. It is used to identify strong edges in the image. 
    Any edge with a gradient magnitude higher than this threshold is considered a strong edge.
    '''

    cv2.imshow('Edge Detection', img_edge)


    cv2.waitKey(0)

def Dilate_and_Erode():
    
    #this examples builds off of the edge detection function above
    
    #Supporting Docs
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    
    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    img_edge = cv2.Canny(img, 100,200)
    
    img_edge_dilate = cv2.dilate(img_edge, np.ones((5,5), dtype = np.int8))
    
    img_edge_erode = cv2.erode((img_edge_dilate, np.ones((5,5), dtype = np.int8)))
    

    cv2.imshow('Original Pic', img)
    cv2.imshow('Edge Detection', img_edge)
    cv2.imshow('Dilate', img_edge_dilate)
    cv2.imshow('Erode', img_edge_erode)

    cv2.waitKey(0)
    
if __name__ == '__main__':
    
    #ReadPicture()
    #ReadVideo()
    #ResizePicture_ParameterWay()
    #ResizePic_EasyWay()
    #CropPic()
    #ConvertColor()
    #GreyScale()
    #General_Blurring()
    #Simple_Thresholding()
    #Adaptive_Thresholding()
    #Edge_Detection()
    #Dilate_and_Erode()
    #change to whatever function you want
    pass
