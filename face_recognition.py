import cv2
import numpy as np
import dlib
import os
import logging
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tkinter import Tk
from tkinter.filedialog import askdirectory

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

def get_greypic(image):
    '''(String) -> image
    Given a string representation of a directory which contains an image
    return a greyscale version of that picture. Function returns none if
    the link is invalid.
    '''
    # convert the image from a colour image to a grey scaled image
    grey_image = cv2.imread(image, 0)
    # check if valid image is given to function
    if grey_image.size != None:
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
    os.chdir(original_path)
    # this is the opencv face detector, the input we use is the grey image
    # given by our greyface function
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # get faces in the image
    faces = classifier.detectMultiScale(grey_image, minNeighbors=15)
    os.chdir(path)
    # if there are no faces in the image then return -1, -1 indicating
    # there are no faces in the picture
    if len(faces) == 0:
        return -1, -1
    # get left, bottom, right and top coordinated of the face
    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    rect = (x, y, w, h)
    # get the part of the grey image which contains the face
    grey_face = grey_image[y:y+h, x:x+w]
    # get the rectangle centre and width and height which contains the face
    # return the grey face and the rectangle
    return grey_face, rect, grey_image

def draw_face_outline(rect, image):
    '''((int, int, int, int), numpy.array) -> img
    Given the dimensions of the rectangle which contains a face, return the
    image with the rectangle which contains the face in the image. 
    '''
    # get x coordinate
    x = rect[0]
    # get y coordinate
    y = rect[1]
    # get the starting coordinate of the rectangle (bottom left)
    start = (x, y)
    # get the width of the rectangle
    w = rect[2]
    # get the height of the rectangle
    h = rect[3]
    # get the ending coordinate of the rectangle (up right)
    end = (x+w, y+h)
    # return the image with the blue rectangle outlining the face
    return cv2.rectangle(image, start, end, (0, 204, 255), 2)

def show_face_outline(file):
    '''(String) -> None
    Given the string representation of the file name, show the rectangle outlining
    the face in the greyscale version of the image. This function returns None
    '''
    face_only, rectangle, grey_image = get_face(file)
    # outline the face with a rectangle
    faceoutline = draw_face_outline(rectangle, grey_image)
    # show the greyscale outlined image
    #show_image(faceoutline)
    # return None
    return face_only, rectangle

def show_image(image):
    '''(Image) -> None
    Given a string representation of a directory which contains an image return
    None, show the user the image given to the show_image function
    '''
    # open the image and resize it to be in a 300x300 window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 300,300)
    cv2.imshow('image', image)
    print("press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def get_images(directory):
    '''(String) -> [String]
    Given a string representation of the directory which contains images of the
    same person used to train data, return the images in the directory
    '''
    # valid picture forms
    valid_files = ["jpg", "jpeg", "png", "bmp"]
    # get files that are in the directory given to the function
    files = [file for file in os.listdir(directory)
             if any(file.endswith(filetype) for filetype in
                valid_files)]
    # return the images
    return files

def get_faces(directory):
    '''(String) -> [np.array], int, int
    Given a string representation of a directory, return the detected faces
    the minimum height of the faces detected, and the minimum width of the
    faces detected
    '''
    # get images from directory
    images = get_images(directory)
    # create empty lists
    detected_faces = []
    min_height = []
    min_widths = []
    # create 0 value for keeping track of number of faces
    samples = 0
    # loop through each image and show face outline and add
    # faces to detected faces as well as the height and width of
    # each image
    for image in images:
        face, rectangle = show_face_outline(image)
        if len(face) != 0:
            detected_faces.append(face)
            height, width = face.shape
            min_height.append(height)
            min_widths.append(width)
            samples += 1
    # if face count is 0 then exit function since no faces found
    if samples == 0:
        print("No faces found in directory")
        exit()
    # return detected faces, and minimum of height and width of face dimension
    return detected_faces, min(min_height), min(min_widths)

def resize_faces(detected_faces, min_height, min_width):
    '''([np.array], int, int) -> [np.array], int, int
    Given an array of images represented as np.arrays, which are the
    detected faces, the minimum height of the faces, and minimum
    width of the faces, return the faces resized according to the
    minimum dimension of the faces in detected_faces
    '''
    # create empty list
    resized_faces = []
    # resize each face to size of smallest dimension face and add it to list of resized faces
    for face in detected_faces:
        resized_face = cv2.resize(face, dsize = (min_height, min_width), interpolation=cv2.INTER_CUBIC)
        resized_faces.append(resized_face)
    # return resized faces, number of faces, and minimum height dimension and minimum width dimension
    return resized_faces, len(resized_faces), min_height, min_width

def getPCAMatrix(resized_faces, num_faces, min_height, min_width):
    '''([np.array], int, int, int) -> np.array
    Given the resized faces, number of faces, minimum height dimension
    and the minimum width dimension of the faces compile the faces into
    one matrix for use in PCA
    '''
    # create single matrix of faces
    PCAMatrix = np.reshape(resized_faces, (num_faces, min_height*min_width))
    # return single matrix
    return PCAMatrix

def createLabels(resized_faces):
    '''([np.array]) -> list
    Given a list of resized faces, get the list of labels by getting the
    names of the people in the images
    '''
    # create empty labels list
    labels = []
    # create list for distinct labels
    distinct_labels = []
    # create blank variable for name
    name = None
    # loop through each face
    for face in resized_faces:
        # show each face
        show_image(face)
        # if there are no labels yet, get the name of the person in the image
        # and add it to the list of labels
        if len(labels) == 0:
            label = input("Please input the name of the person in the image: ")
            labels.append(label)
            # add name to distinct labels list
            distinct_labels.append(label)
        # if there are names in the labels list already
        else:
            # create count and iterate through faces giving numbered list
            # of already existing people in the labels
            count = 1
            for i in distinct_labels:
                print('%d. %s' %(count, i))
                count += 1
            # print 0 option for adding a new person
            print("0. add new person")
            # prompt user input
            label = input("Please input the number which corresponds to person in image: ")
            # if user enters 0 get name of person in image and add it to labels list
            if label == "0":
                name = input("Type the name of the person who is in the image: ")
                labels.append(name)
                # add name if it doesn't exist already in the labels list
                if name not in distinct_labels:
                    distinct_labels.append(name)
            # if invalid entry is given, ask for valid input
            elif (int(label) > count) or (int(label) < 0):
                print("Please try a valid input")
            # else 
            else:
                # get the name from the distict label list
                index = int(label)-1
                new_label = distinct_labels[index]
                # add entered label into label list
                labels.append(new_label)
                # if the new label isnt in the distinct labels list
                if new_label not in distinct_labels:
                    # add it to the distinct labels list
                    distinct_labels.append(new_label)
    # return labels list
    return labels

def eigenfaces(directory, labels):
    '''(String, List) -> np.array, list
    Given a directory and a list of labels get the predictions and true
    labels.
    '''
    # get the detected faces
    detected_faces, min_height, min_width = get_faces(directory)
    resized_faces, num_faces, height, width = resize_faces(detected_faces, min_height, min_width)
    PCAMatrix = getPCAMatrix(resized_faces, num_faces, height, width)
    # createLabels function can be used to create your own labels for your dataset if no labels are
    # given
    if labels==None:
        labels = createLabels(resized_faces)
    # create training and test sets and labels. 20% of the data will be the test data
    train_x, test_x, train_labels, test_labels = train_test_split(PCAMatrix, labels, test_size = 0.2)
    # we don't have many pictures so we will set the number of components in PCA to 4
    components = 10
    # create a PCA model with 4 components
    pca = PCA(components)
    # fit the PCA Model with the training data
    pca.fit(train_x)
    # get components for each training face
    train_x_PCA = pca.transform(train_x)
    # get components for each test face
    test_x_PCA = pca.transform(test_x)
    # create neural network
    classifier = MLPClassifier(hidden_layer_sizes=(1024))
    # fit classifier with labels and PCA components for training data
    classifier.fit(train_x_PCA, train_labels)
    # get predictions for test data
    predictions = classifier.predict(test_x_PCA)
    # get predictions and labels
    return predictions, test_labels

def getPerson(directory, labels, testDir):
    '''(String, [String], String) -> [String]
    Given a directory for the training data, [an optional list of labels for training data], and the
    directory for the image which needs identification, return the name of the person in the picture
    which requires identification.
    '''
    # get the detected faces
    detected_faces, min_height, min_width = get_faces(directory)  
    # set the cwd as the folder of the test picture
    os.chdir(testDir)
    #get the face from the supplied image
    detected_face, height, width = get_faces(testDir)
    # if height or width of the test image is smaller than the training images set the min_height
    # and min_width as that of the test image
    if height < min_height:
        min_height = height
    if width < min_width:
        min_width = width
    # resize the train faces
    resized_faces, num_faces, height, width = resize_faces(detected_faces, min_height, min_width)
    # createLabels function can be used to create your own labels for your dataset if no labels are
    # given
    if labels==None:
        labels = createLabels(resized_faces)  
    # resize the test image to match the dimensions of the smallest training image
    resized_face, num_face, h, w = resize_faces(detected_face, min_height, min_width)
    # get the PCA matrix for the training images
    PCAMatrix = getPCAMatrix(resized_faces, num_faces, height, width)
    # get the PCA matrix for the test image
    PCAMatrixtest = getPCAMatrix(resized_face, 1, height, width)
    # Get the PCA matrix for the training data and their corresponding labels
    # test size is set to 0.1 because that is the minimum value we can set to get the ordered labels
    # corresponding to the PCA matrix    
    train_x, test_x, train_labels, test_labels = train_test_split(PCAMatrix, labels, test_size = 0.1)  
    # we don't have many pictures so we will set the number of components in PCA to 4
    components = 12
    # create a PCA model with 4 components
    pca = PCA(components)
    # fit the PCA Model with the training data
    pca.fit(PCAMatrix)
    # get components for each training face
    train_x_PCA = pca.transform(train_x)    
    # get components for the test face
    test_x_PCA = pca.transform(PCAMatrixtest)
    # initialize the classifier
    classifier = MLPClassifier(hidden_layer_sizes=(1024))
    # fit classifier with labels and PCA components for training data
    classifier.fit(train_x_PCA, train_labels)
    # get predictions for test picture
    predictions = classifier.predict(test_x_PCA)    
    # give the prediction for the person in the test image
    return predictions, resized_face
    
    
    

if __name__ == "__main__":
    
    # Get the folder containing training images
    path = askdirectory(title='Select Folder Containing Training Images')
    # get the original path of the directory containing the python file
    original_path = os.getcwd()
    # set the cwd as that of the folder containing training images
    os.chdir(path)
    # get the directory of the test image. THIS MUST BE DIFFERENT THAN THE ONE OF THE TRAINING IMAGES
    testDir = askdirectory(title='Select Folder Containing Test Image')
    # set labels to a list of names in the order of the pictures asked by the script to speed up labelling process
    labels = ["The Weeknd", "Michael Jackson", "The Weeknd", "Michael Jackson","The Weeknd", "Michael Jackson","Michael Jackson","Michael Jackson","The Weeknd","The Weeknd","The Weeknd","The Weeknd"]
    #labels = None
    Tk().destroy()
    #### UN COMMENT TO GET 20% TESTING DATA PREDICATIONS
    #predictions, true_labels = eigenfaces(path, labels)
    
    #get the name of the person who you want to predict
    predictions, face = getPerson(path, labels, testDir)
    plt.ion()
    plt.figure()
    plt.imshow(face[0], interpolation='nearest')
    plt.draw()
    # print the name of the person you want to predict
    print("The name of the person in your picture is " + predictions[0])
    
    