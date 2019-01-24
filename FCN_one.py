from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as scio
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import BatchDatsetReader as dataset
import cv2



#keras

from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf








FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for validation")
tf.flags.DEFINE_string("log", "logs/psp/log/", "path to logs directory")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("image_logs_dir", "logs\\output", "path to image logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = 2
IMAGE_SIZE = 64
IMAGE_SIZE_r = 2000
IMAGE_SIZE_c = 2000




def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) 
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev) 
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)








def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config
def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (2000, 2000):
        kernel_strides_map = {1: 1, # (473-60)/60 + 1 = 6 + 1 = 7（采用的不添加全为零 向上取整  =[(in-filter+1)/stride]）
                              2: 2, # (473-30)/30 + 1 = 11 + 1 = 12
                              3: 3, # (473-20)/20 + 1 = 22 + 1 = 23
                              6: 4} # (473-10)/10 + 1 = 46 + 1 = 47
    elif input_shape == (713, 713):  
        kernel_strides_map = {1: 90,  # (713-90)/90 + 1 = 6 + 1 = 7
                              2: 45,  # (713-45)/45 + 1 = 14 + 1 = 15
                              3: 30,  # (713-30)/30 + 1 = 6 + 1 = 23
                              6: 15}  # (713-15)/15 + 1 = 6 + 1 = 47
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        # exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])     #每个池化核大小
    strides = (kernel_strides_map[level], kernel_strides_map[level])    #池化步长
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)  #平均池化采用的不添加全为零
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],     #采用1x1卷积降维
                        use_bias=False)(prev_layer)                     #通道降到原本的1/N = 1/4
    prev_layer = BN(name=names[1])(prev_layer)                          #训练数据集进行归一化的操作
    prev_layer = Activation('relu')(prev_layer)                         #relu激活
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)                  #feature_map_size=feature_map_shape  上采样双线性差值
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 16.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))
   # 创建不同尺度的feature
    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res















def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        #pool5 = utils.max_pool_2x2(conv_final_layer)


        psp = build_pyramid_pooling_module(conv_final_layer, (2000, 2000))
        
        psp2 = build_pyramid_pooling_module(conv_final_layer, (2000, 2000))
        
        psp3 = build_pyramid_pooling_module(conv_final_layer, (2000, 2000))
        psp4 = build_pyramid_pooling_module(conv_final_layer, (2000, 2000))
        psp5 = build_pyramid_pooling_module(conv_final_layer, (2000, 2000))
        
        
        psp6 = tf.add(psp, psp2)
        psp6 = tf.add(psp6,psp3)
        psp6 = tf.add(psp6,psp4)
        psp6 = tf.add(psp6,psp5)
        pool5 = utils.max_pool_2x2(psp6)#减小一半






        W6 = utils.weight_variable([7, 7, 2560, 2560], name="W6")
        b6 = utils.bias_variable([2560], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)




        W7 = utils.weight_variable([1, 1, 2560, 2560], name="W7")
        b7 = utils.bias_variable([2560], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)



       



        W8 = utils.weight_variable([1, 1, 2560, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")




        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def main(argv=None):
    #Create placeholders：keep_probability， image， annotation
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_r, IMAGE_SIZE_c, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE_r, IMAGE_SIZE_c, 1], name="annotation")

    #Prediction
    pred_annotation, logits = inference(image, keep_probability)

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    
    print(len(train_records))
    print(len(valid_records))


    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    
    #read dataset of validation    
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    
    #load model
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
        

    if FLAGS.mode == "visualize":
    
        #valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        #filePath_an = "E:\\data\\Xshell\\test\\annotations\\row24col78.png"
        #filePath_im = "E:\\data\\Xshell\\test\\images\\row24col78.png"
        filePath_GF = "./logs/input/1000/row2col3-2.png"              # read GF1 groundTruth.png
        filePath_GF_gt = "./logs/input/1000/row2col3-1.png"  # read GF1 image.png
        
        valid_images = cv2.imread(filePath_GF, -1)
        valid_annotations = cv2.imread(filePath_GF_gt, -1)         
        
        valid_images = valid_images[np.newaxis, :]
        valid_annotations = valid_annotations[np.newaxis, :]
        valid_annotations = valid_annotations[:, :, :, np.newaxis]
        
        valid_annotations = valid_annotations/255 #0-1
        #Accuracy on validation 
        valid_acc = tf.reduce_mean(tf.cast(tf.equal(pred_annotation, valid_annotations), tf.float32))
         
        pred, valid_acc = sess.run([pred_annotation, valid_acc], feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        
        print('Accuracy on valication: ' + str(valid_acc))   

        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
        
        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint16), FLAGS.image_logs_dir, name="inp_" + str(1+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint16), FLAGS.image_logs_dir, name="gt_" + str(1+itr))
            utils.save_image(pred[itr].astype(np.uint16), FLAGS.image_logs_dir, name="pred_" + str(1+itr))

            # scio.savemat(FLAGS.image_logs_dir, {'data': valid_images[itr]})

            print("Saved image: %d" % itr)
    
if __name__ == "__main__":
    tf.app.run()
