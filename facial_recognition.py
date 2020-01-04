import cv2
import numpy as np
import os

def get_greypic(image):
    '''(String) -> image
    Given a string representation of a directory which contains an image
    return a greyscale version of that picture. Function returns none if
    the link is invalid.
    '''
    # convert the image from a colour image to a grey scaled image
    grey_image = cv2.imread(image, 0)
    # check if valid image is given to function
    if grey_image.size != 0:
        # return the greyscale image
        return grey_image
    # else return None
    return None

def get_face(image):
    '''(String) -> image, (int, int, int, int)
    Given a string representation of a directory which contains an image return
    greyscale version of the face in the picture and the rectangle which
    contains the face. Returns -1,-1 if there is no face in the image or the 
    directory is invalid and None, None if the image is invalid.
    '''
    # get greyscale image
    grey_image = get_greypic(image)
    if grey_image.size == 0:
        return None, None
    # this is the opencv face detector, the input we use is the grey image
    # given by our greyface function
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # get faces in the image
    faces = classifier.detectMultiScale(grey_image, minNeighbors=5)
    # if there are no faces in the image then return -1, -1 indicating
    # there are no faces in the picture
    if len(faces) == 0:
        return -1, -1
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    rect = (x, y, w, h)
    # get the part of the grey image which contains the face
    grey_face = grey_image[y: y+h, x: x+w]
    # get the rectangle centre and width and height which contains the face
    # return the grey face and the rectangle
    return grey_face, rect

def draw_face_outline(rect, image):
    '''((int, int, int, int), numpy.array) -> None
    Draw the rectangle which contains the face in the image. 
    '''
    x = rect(0)
    y = rect(1)
    
    cv2.rectangle(image = image, start_po
    
    

def show_image(image):
    '''(Image) -> None
    Given a string representation of a directory which contains an image return
    None, show the user the image given to the show_image function
    '''
    # open the image
    cv2.imshow('image', image)
    print("press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def get_images(directory):
    '''(String) -> None
    Given a string representation of the directory which contains images of the
    same person used to train data.
    '''
