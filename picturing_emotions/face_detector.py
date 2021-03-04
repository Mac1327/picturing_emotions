#from cv2 import imread
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

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

    def get_faces(self, save=False):
    
        picture = plt.imread(self.img)
        results = self.detect()

        faces = []
        for i in range(len(results)):
            # get coordinates
            x1, y1, width, height = results[i]['box']
            x2, y2 = x1 + width, y1 + height 
            # detected face
            face = picture[y1:y2, x1:x2]
            # define subplot
            # why not add the original lable??
            # why not add predicted lable as well??
            plt.subplot(1, len(results), i+1)
            plt.axis('off')
            plt.imshow(face)

            faces.append(face)

        plt.show()

        if save:    
            return faces

      
