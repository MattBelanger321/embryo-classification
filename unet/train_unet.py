import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD
import parse_training_csv as parser

from sklearn.model_selection import train_test_split

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

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)  # Use softmax for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_unet(input, labels, model):
    input = np.asarray(input).reshape(-1,256,256,1)
    labels = np.asarray(labels).reshape(-1,256,256,3)
    # Split labeled data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.2, random_state=42)

    # Fit model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))

    return model

def load_data_and_train(csv_file_path='./data/train.csv', width=256, height=256):
    sample_counter = 1

    model = define_unet()

    # Parse the segmentation masks and original files
    preprocessed_labels = []
    input = [] # initalize an empty list to hold inputs
    for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):
        labels = []  # Initialize an empty list to hold labels
        for (case_name, day, slice_idx, class_name, matrix) in segmentation_mask.values():
            labels.append(matrix)  # Append the matrix to the labels list

        # Convert the labels list to a NumPy array
        labels_array = np.asarray(labels)
        # Reshape labels_array to (height, width, channels) for OpenCV
        labels_array = np.transpose(labels_array, (1, 2, 0))
        labels_array = cv2.resize(labels_array, (width,height), interpolation=cv2.INTER_LINEAR)
        preprocessed_labels.append(labels_array)

        input_as_matrix = cv2.imread(original_file, cv2.IMREAD_GRAYSCALE)/255.0 # re-scale to (0,1)
        input_as_matrix = cv2.resize(input_as_matrix,(width,height), interpolation=cv2.INTER_LINEAR)
        input_as_matrix = np.asarray(input_as_matrix)
        input.append(input_as_matrix)

        if sample_counter % 100 == 0:
            print(f"Training #{1 + sample_counter // 100}")
            model = train_unet(input, preprocessed_labels, model)
            preprocessed_labels = []  # reinitialize to an empty list to hold labels
            input = [] # reinitalize to an empty list to hold inputs
        
        sample_counter += 1

load_data_and_train()