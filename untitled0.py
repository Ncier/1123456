from __future__ import print_function
import tensorflow as tf
import os

import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "Data_zoo\\MIT_SceneParsing\\", "path to dataset")
tf.flags.DEFINE_string("image_dir", "G:\\绿潮数据集\\绿潮数据集2\\80x80", "path to dataset")





def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['images', 'annotations']

    image_list = {}
  
    for directory in directories:
        file_list = []

        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, "training", '*.' + 'png')#Data_zoo\MIT_SceneParsing\ADEChallengeData2016\images\training\*.png
        file_list.extend(glob.glob(file_glob))
        
        
        for f in file_list: 
            image_list[directory].append(f)               
        print ('No. of %s files: %d' % (directory, len(image_list[directory])))                                         
    images = image_list['images']
    masks = image_list['annotations']
           
                
    return images,masks

images,masks=create_image_lists(FLAGS.image_dir)















