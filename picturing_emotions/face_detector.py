#from cv2 import imread
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import tensorflow as tf

class DetectFace:

    def __init__(self, img):
        self.img = img


    def detect(self):
        '''
        Returns results
        It's a list of all different detected faces
        For every face we have dictionary:
        - box: [starting x, starting y, width, height]
        - confidence score between 0-1,
        - keypoints: coordinates for left_eye, right_eye, nose, mouth_left, mouth_right

        '''

        detector = MTCNN()
        pixels = plt.imread(self.img)
        results = detector.detect_faces(pixels)

        return results

    def draw_boxes(self, with_dots=False):
    
        picture = plt.imread(self.img)
        plt.imshow(picture)

        results = self.detect()

        ax = plt.gca()
        for result in results:
            # get coordinates
            x, y, width, height = result['box']
            # create the shape
            rect = Rectangle((x, y), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)

            if with_dots:
                for value in result['keypoints'].values():
                    # create and draw dot
                    dot = Circle(value, radius=2, color='red')
                    ax.add_patch(dot)
        plt.show()

    def get_faces(self, save=True):
    
        picture = plt.imread(self.img)
        results = self.detect()

        faces = []
        X1, X2, Y1, Y2 = [],[],[],[] #<<<< added this to be abel get the location of all the boxes
        for i in range(len(results)):
            # get coordinates
            x1, y1, width, height = results[i]['box']
            x2, y2 = x1 + width, y1 + height 
            # detected face
            face = picture[y1:y2, x1:x2]
            # define subplot
            # why not add the original lable??
            # why not add predicted lable as well?? 
            #plt.subplot(1, len(results), i+1) <<<<<< I commented these lines out to only get the images and not plot it
            #plt.axis('off')  <<<<<
            #plt.imshow(face)  <<<<<<
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

            #faces.append(face)
            faces.append(np.expand_dims(tf.image.resize(face, [224, 224]),axis=0)/255.0)

        #plt.show() <<<<<<


        if save:    
            return faces, X1, X2, Y1, Y2

      
