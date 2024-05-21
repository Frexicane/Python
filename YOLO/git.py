import cv2 as cv  # this is the computer vision header
import time  # time header
import numpy as np  # matrix header


def ReadPicture():
    img = cv.imread('DogZiller.jpg')
    # this reads the image. Put the image name in single quotes

    if img is None:
        print("Error: Unable to read the image file")
        return

    cv.imshow('DogZiller', img)
    # this shows the image
    #this name "DogZiller" will be the title of the pop up window

    cv.waitKey(0)
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

        cv.imshow('Dogzilla_Video', frame)
        # This will show the video. Can replace 'Dogzilla_Video' with '0' for webcam.
        #this name "DogZilla_Video" will be the title of the pop up window


        if cv.waitKey(20) & 0xFF == ord('d'):
            # If the letter 'd' is pressed, end the video.
            break

    capture.release()
    cv.destroyAllWindows()
    # Closes the video window

def ResizePicture(frame, scale = 0.75):
    #takes in the frame (picture) and scales it by 0.75
    #Im just using 0.75 as an example

    img = cv.imread('DogZiller.jpg') #everything from here

    if img is None:
        print("Error: Unable to read the image file")
        return

    cv.imshow('Title', img) # to here need to be OUTSIDE and BEFORE this resize function 

    width = int(frame.shape[1]* scale)
    height = int(frame.shape[1]*scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    cv.waitKey(0)


if __name__ == '__main__':
    #ResizePicture()
    #ReadPicture()
    #ReadVideo()
    #change to whatever function you want
