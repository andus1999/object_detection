import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
from picamera import PiCamera

_camera = PiCamera()
_camera.resolution = (320, 320)

_interpreter = tflite.Interpreter(model_path="model.tflite")
_interpreter.allocate_tensors()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()


def detect(image):
    """
    Detects objects in an Image
        
        Parameters:
            image (PIL.Image): Input image of size 320 x 320
        
        Returns:
            detection_boxes (numpy.array): Array of detection boxes with shape (25, 4)
                A detection box contains the pixel values [ymin, xmin, ymax, xmax].
                
            detected_classes (List[str]): List with labels of detected objects
            
            detection_score (numpy.array): List with scores of detected objects
                A higher score indicates that the prediction is more certain.
    """
    
    input_array = np.array([np.array(image)])
    _interpreter.set_tensor(_input_details[0]['index'], input_array)
    
    _interpreter.invoke()
    
    boxes = _interpreter.get_tensor(_output_details[0]['index'])[0] * 320
    classes = _interpreter.get_tensor(_output_details[1]['index'])[0]
    scores = _interpreter.get_tensor(_output_details[2]['index'])[0]
    num_detections = _interpreter.get_tensor(_output_details[3]['index'])[0]
    
    with open('labelmap.txt', 'r') as f:
        labels = f.read().split('\n')
    str_classes = [labels[int(s)] for s in classes]
    
    return boxes, str_classes, scores


def show_image(min_score=0.2):
    """
    Shows an image with detected objects of the pi camera.
    
        Parameters:
            min_score (float): Float between 0 and 1 indicating the minimum certainty for a box to be drawn.
    """
    
    _camera.capture('frame.jpg')
    image = Image.open('frame.jpg')
    
    boxes, classes, scores = detect(image)
    input_array = np.array([np.array(image)])
    
    fig, ax = plt.subplots(1)
    
    ax.imshow(input_array[0])

    detection_boxes = [patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0])
                       for box in boxes[scores > min_score]]
    patch_collection = PatchCollection(detection_boxes, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_collection(patch_collection)

    for i in range(len(boxes[scores > min_score])):
        box = boxes[i]
        label = classes[i]
        score = scores[i]
        ax.text(box[1], box[2], f'{label} {score:.2f}', color='w')

    plt.show()

