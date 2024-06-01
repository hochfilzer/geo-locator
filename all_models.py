import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from tensorflow import keras
import tensorflow.compat.v1 as tf1
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf1.logging.set_verbosity(tf1.logging.ERROR)
tf1.disable_v2_behavior( )
tf1.enable_eager_execution()




import numpy as np
from pathlib import Path
from scipy.special import softmax


import pandas as pd
from tqdm import tqdm
from glob import glob
import cv2
import torch
from torch.utils import data
from PIL import Image
import torchvision
from torchvision import transforms
import six.moves.urllib as urllib
import tarfile
from collections import Counter



# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
model_path =  "models"
PATH_TO_CKPT = os.path.join(str(Path(__file__).parent), model_path, MODEL_NAME, 'frozen_inference_graph.pb')
TRAFFIC_MODEL_PATH= os.path.join(str(Path(__file__).parent), 'models', 'train1000.pkl')


class FeatureFinder(object):

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #set up device for torch depending on whether a gpu is detected or not

        self.detection_graph = self.load_graph()  #detection graph for coco
        self.extract_graph_components()

        self.sess = tf1.Session(graph=self.detection_graph) #Session is to evaluate the model on coco

        self.model = self.load_the_sign_model() #model is to evaluate using traffic sign model

        # run the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image,0.1,10)
        self.classified_index = 0

        dummy_image = np.transpose(dummy_image, (2, 0, 1)).astype(float) #the traffic sign model needs a tensor in a different shape than coco
        dummy_image=torch.tensor(dummy_image, dtype=torch.float)
        with torch.no_grad():
            self.model([dummy_image.to(self.device)])

    ##################################################
    #Load the traffic sign model
    def load_the_sign_model(self):
        model_path = TRAFFIC_MODEL_PATH
        if not os.path.exists(model_path):
            print('download the model from Kaggle at : https://www.kaggle.com/datasets/bezemekz/traffic-sign-detection-with-faster-r-cnn-pkl or recheck your TRAFFIC_MODEL_PATH')
        else:
            model = torch.load(model_path, map_location=self.device)
            model.to(self.device)
            return model
#####################################################################
#Download the coco model
    def download_model(self):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, os.path.join(str(Path(__file__).parent), 'models', MODEL_FILE))
        tar_file = tarfile.open(os.path.join(str(Path(__file__).parent), 'models', MODEL_FILE))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.path.join(str(Path(__file__).parent), 'models'))
################################
#setting the model: chose of the features to detect
    def load_graph(self):
        if not os.path.exists(PATH_TO_CKPT):
            self.download_model()

        detection_graph = tf1.Graph()
        with detection_graph.as_default():
            od_graph_def = tf1.GraphDef()
            with tf1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf1.import_graph_def(od_graph_def, name='')

        return detection_graph
#######################################################
#selects coco boxes and scores above the threshold
    def select_boxes(self,boxes, classes, scores, score_threshold, target_class):
        """

        :param boxes:
        :param classes:
        :param scores:
        :param target_class: default traffic light id in COCO dataset is 10
        :return:
        """

        sq_scores = np.squeeze(scores)
        sq_classes = np.squeeze(classes)
        sq_boxes = np.squeeze(boxes)

        sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)

        return sq_boxes[sel_id], sq_scores[sel_id]

#########################################################
    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


 #############################################################
    def detect_multi_object(self, image_np, score_threshold,target_class):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sel_boxes,sel_scores = self.select_boxes(boxes=boxes, classes=classes, scores=scores,
                                 score_threshold=score_threshold, target_class=target_class)


        return sel_boxes,sel_scores
##################################################################
    def crop_roi_image(self,image_np, score_threshold,target_class):
        im_height, im_width, _ = image_np.shape
        boxes,scores = self.detect_multi_object(image_np, score_threshold,target_class)
        cropped_images=[]
        for sel_box in boxes:
            (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)
            cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
            cropped_images.append(cropped_image)
        return {'images':cropped_images,'scores':scores}

####################################################################
    def sign_crop(self,file,sign_threshold):
    # The img entered is a tensor in the 0-1 range

        img = cv2.imread(file)

      # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
        img /= 255.0
            # bring color channels to front
        img = np.transpose(img, (2, 0, 1)).astype(float)
                # convert to tensor
        img = torch.tensor(img, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            '''
            prediction Like:
            [{'boxes': tensor([[1221.7869,  523.7036, 1272.7373,  575.1018],
            [ 192.8189,  527.5751,  240.7135,  589.8405],
            [ 197.3745,  538.7914,  235.9153,  572.1550],
            [ 195.1216,  533.9565,  238.6585,  578.0548],
            [ 194.0861,  517.0943,  238.0777,  582.4178]], device='cuda:0'),
            'labels': tensor([7, 7, 7, 8, 5], device='cuda:0'),
            'scores': tensor([0.9792, 0.9036, 0.2619, 0.2407, 0.0575], device='cuda:0')}]
            '''
            signprediction = self.model([img.to(self.device)])

        b = signprediction[0]['boxes']
        #print(b)
        s = signprediction[0]['scores']
        #print(s)

        #Apply Non-maximum suppression:
        keep = torchvision.ops.nms(b,s,0.1)
        #print(keep)
        s = [score.item() for score in s]
        if len(s)!=0:
            s=s[:len(keep)]

        img = img.permute(1,2,0)  # C,H,W_H,W,C, for drawing
        img = (img * 255).byte().data.cpu()  # * 255, float to 0-255
        img = np.array(img)  # tensor → ndarray
        #Convert np array img to right format.
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Class number coressponding to Classes


        cropped_im_set=[]
        for k in range(len(keep)):
            xmin = round(signprediction[0]['boxes'][k][0].item())
            ymin = round(signprediction[0]['boxes'][k][1].item())
            xmax = round(signprediction[0]['boxes'][k][2].item())
            ymax = round(signprediction[0]['boxes'][k][3].item())


            cropped_img=img[ymin:ymax,xmin:xmax]
            cropped_im_set.append(cropped_img)

        return {'images':cropped_im_set,'scores':s}
 ##################################################################
 #this is the function that will actually be used. Imput a file path string as img and a score threshold. Output is a dictionary with keys ['car','bus','lights','stop_sign','signs'].
 # Each value is further dictionary with keys 'images' and 'scores'. The 2 values in each of these dictionaries are lists of np array images and their corresponding scores from the two detectors
    def crop_images(self,img, score_threshold):
        image_np=np.asarray(Image.open(img))
        target_classes=[3,4, 6,10,13] #car,bus,traffic lights, stop sign
        class_names=['cars', 'motorcycles' ,'buses','lights','stop_sign']
        images_dict={class_names[k]:self.crop_roi_image(image_np,score_threshold,target_class) for k, target_class in enumerate(target_classes)}
        signs_dict={'signs':self.sign_crop(img,score_threshold)}
        images_dict =images_dict|signs_dict
        return images_dict






# This is a list with the potential features that were extracted
# features = ['car','bus','lights','stop_sign', 'signs']
features = ['cars','buses','lights','stop_sign', 'signs', 'motorcycles']
cities = ['Madrid', 'Phoenix', 'Miami', 'Boston', 'Brussels', 'Rome', 'Barcelona', 'Chicago', 'Lisbon', 'Melbourne', 'Minneapolis', 'Bangkok', 'TRT', 'London', 'PRG', 'Osaka', 'PRS']

# Fill this up with the remaining links
model_paths = {'stop_sign':  os.path.join(str(Path(__file__).parent), 'models','Vanilla_CNN_feature_classifier_5_epochs_COCO_stop_signs_BALANCED_dataset_29052024.keras'),
                'motorcycles': os.path.join(str(Path(__file__).parent), 'models', 'Vanilla_CNN_feature_classifier_5_epochs_COCO_motorcycles_BALANCED_dataset_30052024.keras'),
                'lights': os.path.join(str(Path(__file__).parent), 'models', 'Vanilla_CNN_feature_classifier_3_epochs_COCO_traffic_lights_BALANCED_dataset_29052024.keras'),
                'signs': os.path.join(str(Path(__file__).parent), 'models', 'GPU_model_classifier_signs_with3km_withpriorityroad_27052024.keras'),
                'buses': os.path.join(str(Path(__file__).parent), 'models', 'Vanilla_CNN_feature_classifier_2_epochs_COCO_buses_BALANCED_dataset_31052024.keras'),
                'cars': os.path.join(str(Path(__file__).parent), 'models', 'Vanilla_CNN_feature_classifier_5_epochs_COCO_cars_BALANCED_dataset_30052024.keras')}

image_dimensions = {'stop_sign': (62,52),
                    'motorcycles': (64,61),
                    'lights': (45,21),
                    #'signs': (49,73),
                    'signs': (34,35),
                    'cars': (49,73),
                    'buses': (156,199)}
global_threshold = 0.6

class Model:
    """
    Class whose main method .predict can predict the city in which a photo was taken in (out of 17 options)

    The class has two parameters:
        feature: str
            This is by default set to None, in which case the end_to_end model is instantiated, otherwise the specific model
            going by feature (e.g. 'signs') is instantiated.
        verbose: int
            Set this to 0 if you want to repress most log messages
    """
    def __init__(self, feature = None):


        self.feature = feature
        if feature:
            self.model = tf.keras.models.load_model(model_paths[feature])
            self.feat_finder = FeatureFinder()
        else:
            self.model = tf.keras.models.load_model(os.path.join(str(Path(__file__).parent), 'models', 'clean_batch_size_128.keras'))




    def preprocess_image_feature(self, img_path, score_threshold):
        """
        Given the path to an image, as well as the feature that is required this function will preprocess
        this image for the trained neural networks.

        img_path: str
        feature: str
        """
        feature = self.feature
        images_dict: Dict[str, List] = self.feat_finder.crop_images(img_path, score_threshold)
        if np.array(images_dict[feature]['scores']).size > 0:
            index = np.argmax(images_dict[feature]['scores'])
            img = images_dict[feature]['images'][index]
            score = images_dict[feature]['scores'][index]
            img = tf.image.resize(
                img,
                image_dimensions[feature],
                method='nearest') # ResizeMethod.NEAREST_NEIGHBOR)

            img = np.array(img)
            img = img.reshape((1,) + img.shape)
            return img, score
        else: # This is the case in which coco does not detect a feature
            return 0, 0
    def preprocess_image_end_to_end(self, img_path):
        """
        Load an image file directly from its path so that it is suitable for the end to end model as trained

        img_path: str
        """
        img = tf.keras.utils.load_img(img_path, target_size = (400,300))
        img = np.array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        # img = np.array(img)
        img = img.reshape((1,) + img.shape)
        return img

    def predict_end_to_end(self, img_path):
        """
        Given the path to an image this function will predict in which city the picture was taken using the end to end model

        img: str, path to the image that we would like to predict
        want_probabilities: bool, If this is set to True then the output is a dictionary of the probabilities for each class.
        """
        global cities
        model = self.model
        img = self.preprocess_image_end_to_end(img_path)
        predictions = softmax(model.predict(img, verbose = 0))
        # cities = ['Madrid', 'Phoenix', 'Miami', 'Boston', 'Brussels', 'Rome', 'Barcelona', 'Chicago', 'Lisbon', 'Melbourne', 'Minneapolis', 'Bangkok', 'TRT', 'London', 'PRG', 'Osaka', 'PRS']
        return {city: predictions[0,i] for i, city in enumerate(cities)}

        # if want_probabilities:
        #     return {city: predictions[0,i] for i, city in enumerate(cities)}
        # else:
        #     return cities[np.argmax(predictions)]

    def predict_feature(self, img_path, score_threshold):

        """
        Given the path to an image this function will predict in which city the picture was taken using the feature model

        img: str, path to the image that we would like to predict
        want_probabilities: bool, If this is set to True then the output is a dictionary of the probabilities for each class.
        """

        global cities
        img, score = self.preprocess_image_feature(img_path, score_threshold)
        if score > 0:
            model = self.model
            predictions = softmax(model.predict(img, verbose = 0))
            # cities = ['Madrid', 'Phoenix', 'Miami', 'Boston', 'Brussels', 'Rome', 'Barcelona', 'Chicago', 'Lisbon', 'Melbourne', 'Minneapolis', 'Bangkok', 'TRT', 'London', 'PRG', 'Osaka', 'PRS']
            return {city: predictions[0,i] for i, city in enumerate(cities)}, score
        else:
            return {city: 0 for city in cities}, 0

    def predict(self, img_path, want_probabilities = False, want_accuracy_score = False, score_threshold = global_threshold):
        """
        The main interesting function for this class. In the default setting returns a predicted class.
        If want_probabilities is set to True then a dictionary with classes and their prediction probabilities is returned.
        If want_accuracy_score is set to True and the model is based on a feature detection mechanism, then the detection
        accuracy score is returned. If this is the end to end model then this setting does not change anything.

        img_path: str
        want_probabilities: bool
        want_accuracy_score: bool
        """
        if self.feature:
            pred, score = self.predict_feature(img_path, score_threshold)
            if want_probabilities and want_accuracy_score:
                return pred, score
            elif want_probabilities:
                return pred
            else:
                return max(pred, key = pred.get)
        else:
            pred = self.predict_end_to_end(img_path)
            if want_probabilities:
                return pred
            else:
                return max(pred, key = pred.get)

class GeoLocator():

    def __init__(self):
        self.models = {"end_to_end": Model()}
        for feature in features:
            self.models[feature] = Model(feature)

    def predict(self, img_path, want_probabilities = False, proportion_end_to_end = 0.8, verbose = 1, score_threshold = global_threshold):

        """
        The main method of this project. This function takes an image and makes a prediction using one end-to-end model and 6 feature
        based models. Arguments:

        img_path: str

        want_probabilities: bool
            Set this to True if you want the output to in the form of a dictionary that assigns each of 17 cities a
            probability of the image being taken in the city.
        proportion_end_to_end: float
            This should be a number between 0 and 1. Determines how much weight is assigned to the end-to-end model
        verbose: int
            Set either 0 or 1. If set to 0 then most output messages will not be displayed.

        """

        predictions = {}
        scores = {}
        for feature in features:
            predictions[feature], scores[feature] = self.models[feature].predict(img_path, want_probabilities = True, want_accuracy_score = True, score_threshold = score_threshold)
        pred_end2end = self.models['end_to_end'].predict(img_path, want_probabilities = True)

        sum_scores = sum(scores.values())
        if sum_scores == 0:
            if verbose == 1:
                print("There were no features found, so the end-to-end model will be used to make a prediction.")
            return max(pred_end2end, key = pred_end2end.get)

        # The following line computes the number of active features
        # number_features = len(features) - Counter(scores.values()).get(0,0)
        number_features = 0
        for feature in features:
            if scores[feature] > 0:
                number_features +=1
                features_for_humans = {'stop_sign': 'a stop sign',
                                        'signs': 'street signs',
                                        'buses': 'a bus',
                                        'cars': 'a car',
                                        'lights': 'traffic lights',
                                        'motorcycles': 'a motorcycle'}
                if verbose == 1:
                    print(f"The COCO detector found {features_for_humans[feature]}!")

        prop_end2end = proportion_end_to_end
        prop_feature = 1 - prop_end2end

        final_pred = {}
        for city in cities:
            feature_contribution = 0
            for feature in features:
                feature_contribution += predictions[feature][city] * scores[feature]

            final_pred[city] = (prop_end2end * pred_end2end[city]
                                + prop_feature/sum_scores * feature_contribution)
        if want_probabilities:
            return final_pred
        else:
            return max(final_pred, key = final_pred.get)
