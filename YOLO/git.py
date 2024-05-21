#import cv2 as cv  # this is the computer vision header for old cv NOT cv2
import cv2 # cv2 header
import time  # time header
import numpy as np  # numpy/matrix header
import os 
from matplotlib import pyplot as plt


def ReadPicture():
    img = cv2.imread('DogZiller.jpg')
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
    capture = cv2.VideoCapture('YOLO\Dogzilla_Video.mp4')
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
    
    Stand_Dev = 3
    
    img_Guas = cv2.GaussianBlur(img, (k_size, k_size), Stand_Dev)
    #3 is The standard deviation along the x and y directions of the Gaussian kernel. 
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
    
    Stand_Dev = 3
    
    img_Guas = cv2.GaussianBlur(img, (k_size, k_size), Stand_Dev)
    #3 is The standard deviation along the x and y directions of the Gaussian kernel. 
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
    
    Lower_Threshold = 80
    Higher_Threshold = 255
    
    ret, thresh = cv2.threshold(img_greyscaled, Lower_Threshold, Higher_Threshold, cv2.THRESH_BINARY)
    #all values below 80 will go to 0 an anything above 80 will go to 255
    
    #this example is using threshold binary but there are many types from the link above
    
    thresh = cv2.blur(thresh,(10,10))                                   #can delete this
    #10 and 10 are the k sizes used (just an example)
    
    ret, thresh = cv2.threshold(thresh, Lower_Threshold, Higher_Threshold, cv2.THRESH_BINARY)     #can delete this
    
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
    
    Lower_Threshold = 21
    Higher_Threshold = 30
    
    thresh = cv2.adaptiveThreshold(img_greyscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,Lower_Threshold,Higher_Threshold)
    
    #the 255 means each pixel will be set to white (0 is black, 255 is white)
    
    #this is using the adaptive GAUSSIAN blue, there are others
    
    #the 21 and 30 are the thresholds for the regions
    
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

    Lower_Threshold = 100
    Higher_Threshold = 200


    img_edge = cv2.Canny(img, Lower_Threshold,Higher_Threshold)
    
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

    Lower_Threshold = 100
    Upper_Threshold = 200

    img_edge = cv2.Canny(img, Lower_Threshold, Upper_Threshold)
    
    img_edge_dilate = cv2.dilate(img_edge, np.ones((5,5), dtype = np.int8))
    
    img_edge_erode = cv2.erode((img_edge_dilate, np.ones((5,5), dtype = np.int8)))
    

    cv2.imshow('Original Pic', img)
    cv2.imshow('Edge Detection', img_edge)
    cv2.imshow('Dilate', img_edge_dilate)
    cv2.imshow('Erode', img_edge_erode)

    cv2.waitKey(0)
    
    
def Add_Line():
    
    img = cv2.imread('Chalkboard.jpg')
    
    print(img.shape)
    #print this to show the size of the pic in the terminal
    #so that way you dont plot a point outside the boundary
    
    #** remember in the terminal it will print (y,x) instead of (x,y)

    
    if img is None:
        print("Error: Unable to read the image file")
        return
    
    Starting_Point_X = 100
    Starting_Point_Y = 150
    Ending_Point_X = 300
    Ending_Point_Y = 350
    R = 0
    G = 0
    B = 255
    Thickness = 3
    
    cv2.line(img,(Starting_Point_X,Starting_Point_Y),(Ending_Point_X, Ending_Point_Y), (R,G,B), Thickness)
    #(R,G,B) values are measured 0-255, so if you want red do (255,0,0)
    #If you want blue, use (0,0,255)
    #If you want a color thats not just PURE red, green, or blue, you have to look up the color combo


    cv2.imshow('Chalkboard', img)

    cv2.waitKey(0)


def Add_Rectangle():

    
    img = cv2.imread('Chalkboard.jpg')
    
    print(img.shape)
    #print this to show the size of the pic in the terminal
    #so that way you dont plot a point outside the boundary
    
    #** remember in the terminal it will print (y,x) instead of (x,y)

    
    if img is None:
        print("Error: Unable to read the image file")
        return
    
    Top_Left_X = 100
    Top_Left_Y = 150
    Bottom_Right_X = 300
    Bottom_Right_Y = 350
    #to make a rectangle you are plotting the top left corner and bottom right 
    R = 0
    G = 0
    B = 255
    Thickness = 3
    #if your thickness is -1, then it will be a solid filled rectangle*****
    
    cv2.rectangle(img,(Top_Left_X, Top_Left_Y),(Bottom_Right_X, Bottom_Right_Y),(R,G,B), Thickness)
    
    cv2.imshow('Rectangle', img)
    
    cv2.waitKey(0)
    
def Add_Circle():

    img = cv2.imread('Chalkboard.jpg')
    
    print(img.shape)
    #print this to show the size of the pic in the terminal
    #so that way you dont plot a point outside the boundary
    
    #** remember in the terminal it will print (y,x) instead of (x,y)
    
    if img is None:
        print("Error: Unable to read the image file")
        return
    
    Center_X = 350
    Center_Y = 350
    Radius = 100
    R = 100
    G = 0
    B = 200
    Thickness = 3
    #if your thickness is -1, then it will be a solid filled *****
    
    cv2.circle(img,(Center_X, Center_Y),Radius, (R,G,B), Thickness)
    
    cv2.imshow('Circle', img)
    
    cv2.waitKey(0)

def Add_Text():

    #supporting documents
    
    #https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html

    img = cv2.imread('Chalkboard.jpg')
    
    print(img.shape)
    #print this to show the size of the pic in the terminal
    #so that way you dont plot a point outside the boundary
    
    #** remember in the terminal it will print (y,x) instead of (x,y)
    
    if img is None:
        print("Error: Unable to read the image file")
        return
    
    X_Position = 50
    Y_Position = 240
    Font = cv2.FONT_HERSHEY_SIMPLEX
    R = 100
    G = 0
    B = 100
    Thickness = 2
    TextSize = 2
    
    cv2.putText(img, 'This is a Text', (X_Position, Y_Position), Font, TextSize,  (R, G, B), Thickness)
    #cv2.putText(img, 'This is a Text', (X_Position, Y_Position), cv2.FONT_HERSHEY_SIMPLEX, (B, G, R), Thickness)

    
    cv2.imshow('Text', img)
    
    cv2.waitKey(0)
    
    
    
def Contours():
    
    #Supporting Docs
    
    #https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

    img = cv2.imread('DogZiller.jpg')

    if img is None:
        print("Error: Unable to read the image file")
        return

    Lower_Bound_Thresh = 125
    Upper_Bound_Thresh = 255
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #need to convert to grey first

    ret, thresh = cv2.threshold(img, Lower_Bound_Thresh, Upper_Bound_Thresh, cv2.THRESH_BINARY_INV)
    #applied an inverse threshold
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        R = 0
        G = 255
        B = 0
        Thickness = 2
        if cv2.contourArea(cnt) > 200:
            #cv2.drawContours(img, cnt, -1 (R,G,B), Thickness)
            
            x1, y1, w, h = cv2.boundingRect(cnt)
            
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (R,G,B), Thickness)
    
    cv2.imshow('Greyed', img)
    cv2.imshow('Contour', thresh)

    cv2.waitKey(0)


if __name__ == '__main__':
    
    ReadPicture()
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
    #Add_Line()
    #Add_Rectangle()
    #Add_Circle()
    #Add_Text()
    #Contours()
    #change to whatever function you want
    pass
    
