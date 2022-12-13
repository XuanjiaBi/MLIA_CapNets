# Copyright 2021  Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import SimpleITK
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import glob
from utils import pre_process_mnist, pre_process_multimnist, pre_process_smallnorb
import json
from keras.models import load_model




class Dataset(object):
    """
    A class used to share common dataset functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    config_path: str
        path configuration file
    
    Methods
    -------
    load_config():
        load configuration file
    get_dataset():
        load the dataset defined by model_name and pre_process it
    get_tf_data():
        get a tf.data.Dataset object of the loaded dataset. 
    """
    def __init__(self, model_name, config_path='config.json'):
        self.model_name = model_name
        self.config_path = config_path
        self.config = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.class_names = None
        self.X_test_patch = None
        self.load_config()
        self.get_dataset()
        

    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)

    def import_mhd(self):
        # Training
        training_diseased_path = r'C:\Users\bixua\Desktop\Efficient-CapsNet\Classification_data\Training\Diseased\*.mhd'
        training_healthy_path = r'C:\Users\iain8\Documents\MLIA\Efficient-CapsNet\Classification_data\Training\Healty\*.mhd'

        diseased_files = glob.glob(training_diseased_path)
        healthy_files = glob.glob(training_healthy_path)

        files = diseased_files + healthy_files
        print("type of files: ",type(files))

        # label one for diseased
        label_one = np.ones(len(diseased_files))
        label_zero = np.zeros(len(healthy_files))
        # print(label_zero)
        # label = np.concatenate(label_one,label_zero)
        train_y = np.hstack((label_one,label_zero))
        # print(label)
        train_x = sitk.GetArrayFromImage(sitk.ReadImage(files))

        # Testing
        testing_diseased_path = r'C:\Users\bixua\Desktop\Efficient-CapsNet\Classification_data\Testing1\Diseased\*.mhd'
        testing_healthy_path = r'C:\Users\bixua\Desktop\Efficient-CapsNet\Classification_data\Testing1\Healthy\*.mhd'

        diseased_files = glob.glob(testing_diseased_path)
        healthy_files = glob.glob(testing_healthy_path)

        files = diseased_files + healthy_files
        print("type of files: ",type(files))

        # label one for diseased
        label_one = np.ones(len(diseased_files))
        label_zero = np.zeros(len(healthy_files))
        # print(label_zero)
        # label = np.concatenate(label_one,label_zero)
        testing_y = np.hstack((label_one,label_zero))
        # print(label)
        testing_x = sitk.GetArrayFromImage(sitk.ReadImage(files))

        return train_x, train_y,testing_x,testing_y

    def get_dataset(self):
        if self.model_name == 'MNIST':
            # self.import_mhd()
            # (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data(path=self.config['mnist_path'])
            self.X_train, self.y_train, self.X_test, self.y_test= self.import_mhd()
            print(self.X_train.shape)
            # self.X_train, self.y_train, self.X_test, self.y_test= self.import_mhd()
            # print(type(self.y_train))
            #print(self.y_train)
            # prepare the data
            self.X_train, self.y_train = pre_process_mnist.pre_process(self.X_train, self.y_train)
            self.X_test, self.y_test = pre_process_mnist.pre_process(self.X_test, self.y_test)
            self.class_names = list(range(10))
            print("[INFO] Dataset loaded!")
        elif self.model_name == 'CUSTOM':
            self.X_train, self.y_train, self.X_test, self.y_test= self.import_mhd()
            # print(type(self.y_train))
            #print(self.y_train)
            # prepare the data
            self.X_train, self.y_train = pre_process_mnist.pre_process(self.X_train, self.y_train)
            self.X_test, self.y_test = pre_process_mnist.pre_process(self.X_test, self.y_test)
            self.class_names = list(range(10))
            print("[INFO] Dataset loaded!")
        elif self.model_name == 'SMALLNORB':
                    # import the datatset
            (ds_train, ds_test), ds_info = tfds.load(
                'smallnorb',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=False,
                with_info=True)
            self.X_train, self.y_train = pre_process_smallnorb.pre_process(ds_train)
            self.X_test, self.y_test = pre_process_smallnorb.pre_process(ds_test)

            self.X_train, self.y_train = pre_process_smallnorb.standardize(self.X_train, self.y_train)
            self.X_train, self.y_train = pre_process_smallnorb.rescale(self.X_train, self.y_train, self.config)
            self.X_test, self.y_test = pre_process_smallnorb.standardize(self.X_test, self.y_test)
            self.X_test, self.y_test = pre_process_smallnorb.rescale(self.X_test, self.y_test, self.config) 
            self.X_test_patch, self.y_test = pre_process_smallnorb.test_patches(self.X_test, self.y_test, self.config)
            self.class_names = ds_info.features['label_category'].names
            print("[INFO] Dataset loaded!")
        elif self.model_name == 'MULTIMNIST':
            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data(path=self.config['mnist_path'])
            # prepare the data
            self.X_train = pre_process_multimnist.pad_dataset(self.X_train, self.config["pad_multimnist"])
            self.X_test = pre_process_multimnist.pad_dataset(self.X_test, self.config["pad_multimnist"])
            self.X_train, self.y_train = pre_process_multimnist.pre_process(self.X_train, self.y_train)
            self.X_test, self.y_test = pre_process_multimnist.pre_process(self.X_test, self.y_test)
            self.class_names = list(range(10))
            print("[INFO] Dataset loaded!")


    def get_tf_data(self):
        if self.model_name == 'MNIST':
            dataset_train, dataset_test = pre_process_mnist.generate_tf_data(self.X_train, self.y_train, self.X_test, self.y_test, self.config['batch_size'])
        elif self.model_name == 'SMALLNORB':
            dataset_train, dataset_test = pre_process_smallnorb.generate_tf_data(self.X_train, self.y_train, self.X_test_patch, self.y_test, self.config['batch_size'])
        elif self.model_name == 'MULTIMNIST':
            dataset_train, dataset_test = pre_process_multimnist.generate_tf_data(self.X_train, self.y_train, self.X_test, self.y_test, self.config['batch_size'], self.config["shift_multimnist"])

        return dataset_train, dataset_test
