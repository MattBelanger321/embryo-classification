from random import shuffle
import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import tensorflow as tf
import glob
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD, Adam
import parse_training_csv as parser
from batch_generator import UNetBatchGenerator as batch_generator

from sklearn.model_selection import train_test_split

import FlushableStream

import numpy as np
import cv2

def define_unet():
    inputs = Input(shape=(256, 256, 1))  # Adjust input shape as needed

	# we use padding same so that the input doesnt change during convolutions
	# the naming convention is
    # c for convolution
    # d for down sampling
    # u for up sammpling
    # the numbers correspond to the position of the group in the U,
    # see section 2 of the UNet paper and figure 1

    # Encoder (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c1)
    d1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c2)
    d2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c3)
    d3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c4)
    d4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck, this is the bottom of the U
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c5)

    # Decoder (upsampling)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])  # Skip connection
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])  # Skip connection
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])  # Skip connection
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])  # Skip connection
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c9)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c9)  # Use softmax for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = Adam( clipnorm=1.0)  # Norm is clipped to 1.0
    model.compile(optimizer=opt, loss='bce', metrics=['accuracy'])
    
    return model

def train_unet(train, test, model, batch_size=32, epochs=10, steps_per_epoch=1203):
    print("Fitting...")
    # Fit model and validate on test data after each epoch

    gen = batch_generator(train, batch_size) # the training data generator

    history = model.fit(gen, epochs=epochs, validation_data=test, verbose=1)
    # Evaluate on the test dataset
    print("Evaluating..")
    _, acc = model.evaluate(test, verbose=1)
    print('Test Accuracy: %.3f' % (acc * 100.0))
    
    return model

# Function to check for NaN values in a batch of input/label pairs
def contains_nan(batch):
    inputs, labels = batch
    # Check if any NaN values are present in the inputs or labels separately
    nan_in_inputs = tf.reduce_any(tf.math.is_nan(inputs))
    nan_in_labels = tf.reduce_any(tf.math.is_nan(labels))
    return nan_in_inputs or nan_in_labels


# Function to check the entire dataset for NaN values
def dataset_has_nan(dataset):
    for batch in dataset:
        if contains_nan(batch):
            print("NaN values found in the dataset!")
            return True
    print("No NaN values found in the dataset.")
    return False


def is_float32(dataset):
    for batch in dataset:
        # Assuming the batch is of the form (inputs, labels)
        inputs, labels = batch
        
        # Check if dtype of inputs and labels is tf.float32
        if inputs.dtype != tf.float32 or labels.dtype != tf.float32:
            print("Data is not Float32")
            return False
            
    print("Data is float32")
    return True

def is_normalized(dataset):
    for batch in dataset:
        # Assuming the dataset is of the form (inputs, labels)
        inputs, labels = batch
        
        # Check if all values in inputs are within the range [0, 1]
        if not tf.reduce_all((inputs >= 0) & (inputs <= 1)):
            print("Inputs are not on [0,1]")
            return False
        
        # Check if all values in labels are within the range [0, 1]
        if not tf.reduce_all((labels >= 0) & (labels <= 1)):
            print("Labels are not on [0,1]")
            return False
            
    print("Data is normalized")
    return True


# This function will check if the dataset is fit for training
def validate_dataset(dataset):
    return not dataset_has_nan(dataset) and is_float32(dataset) and is_normalized(dataset)

def get_dataset(input_dir, label_dir, batch_size):
    # Get list of all input and label files
    input_filenames = glob.glob(f"{input_dir}/*.npy")
    label_filenames = glob.glob(f"{label_dir}/*.npy")

    # Pair the filenames together
    paired_filenames = list(zip(input_filenames, label_filenames))

    # Shuffle the pairs together
    random.seed(42)
    random.shuffle(paired_filenames)

    # Unzip the shuffled pairs back into separate lists
    input_filenames, label_filenames = zip(*paired_filenames)

    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((list(input_filenames), list(label_filenames)))
    
    # Define a function to load and preprocess each pair of input/label npy files
    def process_npy_file(input_file, label_file):
        # Load the .npy files using numpy_function and cast them to float32
        input_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[input_file], Tout=tf.float32)
        label_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[label_file], Tout=tf.float32)
        
        # Remove any singleton dimensions (e.g., [1, 256, 256, 1] -> [256, 256, 1])
        input_data = tf.squeeze(input_data, axis=0)  # Remove the unnecessary first dimension
        label_data = tf.squeeze(label_data, axis=0)  # Remove the unnecessary first dimension
        
        # Explicitly set the shapes to ensure TensorFlow knows what to expect
        input_data.set_shape([256, 256, 1])  # Assuming input is 256x256 grayscale images
        label_data.set_shape([256, 256, 3])  # Assuming label is 256x256 with 3 classes

        return input_data, label_data

    # Shuffle dataset (you can adjust the buffer size based on your total data)
    # dataset = dataset.shuffle(buffer_size=len(input_filenames), seed=10)

    # Map the file-loading function to the dataset
    dataset_files = dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # add filenames for the generator for debugging
    # (file, filename)
    dataset = tf.data.Dataset.zip((dataset_files, dataset))

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for optimal performance during training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # if not validate_dataset(dataset):
    #     raise Exception("Data is INVALID")

    return dataset, len(input_filenames)

def split_dataset(dataset, dataset_size, split_ratio=0.2):
    test_size = int(split_ratio * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    
    return train_dataset, test_dataset

# Usage
flushable_stream = FlushableStream.FlushableStream("output.log", flush_interval=2)  # Flush every 2 seconds
sys.stdout = flushable_stream  # Redirect stdout

# Example usage:
input_dir = './preprocessed_data2d/input_data'
label_dir = './preprocessed_data2d/labels'
batch_size = 32

# Load and split dataset
dataset, size = get_dataset(input_dir, label_dir, batch_size)

train_dataset, test_dataset = split_dataset(dataset, size)
# Define and train U-Net model
model = define_unet()
model = train_unet(train_dataset, test_dataset, model, batch_size=32)
model.save("model.h5")