import tensorflow as tf
import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


from crnn.crnn_model import crnn_model
from crnn.global_configuration import config
from crnn.local_utils import log_utils, data_utils

from east.icdar import get_images

logger = log_utils.init_logger()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class crnnclass(object):

    def __init__(self):
        self.image_dir = 'data/output/EAST/'
        self.checkpoint_dir = 'model/checkpoint/crnn'
        self.num_classes = 37

    def get_images(self):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(self.image_dir):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(files)))
        return files

    def crnn_detect(self, is_vis: bool=True):
        """
        :param image_path:
        :param weights_path:
        :param is_vis:
        :param num_classes:
        """

        w, h = config.cfg.ARCH.INPUT_SIZE
        inputdata = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 3], name='input')

        codec = data_utils.TextFeatureIO()
        self.num_classes = len(codec.reader.char_dict) + 1 if self.num_classes == 0 else self.num_classes

        net = crnn_model.ShadowNet(phase='Test',
                                   hidden_nums=config.cfg.ARCH.HIDDEN_UNITS,
                                   layers_nums=config.cfg.ARCH.HIDDEN_LAYERS,
                                   num_classes=self.num_classes)

        with tf.variable_scope('shadow'):
            net_out = net.build_shadownet(inputdata=inputdata)

        decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=config.cfg.ARCH.SEQ_LENGTH*np.ones(1),
                                                   merge_repeated=False)

        # config tf session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
        #sess_config.allow_soft_placement = True
        #sess_config.log_device_placement = True

        # config tf saver
        saver = tf.train.Saver()

        sess = tf.Session(config=sess_config)

        with sess.as_default():

            files = self.get_images()

            for image_path in files:
                print(image_path)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, tuple(config.cfg.ARCH.INPUT_SIZE))
                image = np.expand_dims(image, axis=0).astype(np.float32)

                saver.restore(sess=sess, save_path=self.checkpoint_dir)

                preds = sess.run(decodes, feed_dict={inputdata: image})

                preds = codec.writer.sparse_tensor_to_str(preds[0])

                logger.info('Predict image {:s} label {:s}'.format(os.path.split(image_path)[1], preds[0]))

                if is_vis:
                    plt.figure('CRNN Model Demo')
                    plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
                    plt.show()

            sess.close()


if __name__ == '__main__':
    mycrnn = crnnclass()
    mycrnn.crnn_detect()