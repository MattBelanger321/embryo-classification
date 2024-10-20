import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import tensorflow as tf
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD
import parse_training_csv as parser

from sklearn.model_selection import train_test_split

import numpy as np

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

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)  # Use softmax for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_unet(train, test, model, batch_size=32, epochs=10):
    print("Fitting...")
    # Fit model and validate on test data after each epoch
    history = model.fit(train, epochs=epochs, validation_data=test, verbose=2)
    # Evaluate on the test dataset
    print("Evaluating..")
    _, acc = model.evaluate(test, verbose=2)
    print('Test Accuracy: %.3f' % (acc * 100.0))
    
    return model

def get_dataset(input_dir, label_dir, batch_size):
    # Get list of all input and label files
    input_files = sorted(glob.glob(f"{input_dir}/*.npy"))
    label_files = sorted(glob.glob(f"{label_dir}/*.npy"))
    
    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    
    # Shuffle dataset (you can adjust the buffer size based on your total data)
    dataset = dataset.shuffle(buffer_size=len(input_files))
    
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



    # Map the file-loading function to the dataset
    dataset = dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for optimal performance during training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(input_files)

def split_dataset(dataset, dataset_size, split_ratio=0.2):
    test_size = int(split_ratio * dataset_size)
    train_size = dataset_size - test_size
    
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    
    return train_dataset, test_dataset


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