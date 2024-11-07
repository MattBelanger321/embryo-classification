import math
from random import shuffle
import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

import unet_3d
from batch_generator import UNetBatchGenerator3D as batch_generator

import tensorflow as tf
import glob
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
import parse_training_csv as parser
from split_data import get_data_list

import numpy as np
import history_visualization as hv

def train_unet(train, validate, test, model, batch_size=32, epochs=2, spe=2, vsteps=1, save_path="model_epoch_{epoch:02d}.keras"):
    print("Fitting...")
    # Fit model and validate on test data after each epoch

    train_gen = batch_generator(train, batch_size) # the training data generator
    validate_gen = batch_generator(validate, batch_size)
    test_gen =  batch_generator(test, batch_size) # the test data generator

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=save_path,   # Path for saving model; {epoch:02d} allows you to save by epoch number
        save_weights_only=False,  # Set to True if you want to save only weights
        save_freq='epoch',        # Save after each epoch
        verbose=1
    )

    # Pass the checkpoint callback to the fit function
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=validate_gen,
        verbose=1,
        steps_per_epoch=spe,
        validation_steps=vsteps,
        callbacks=[checkpoint_callback]
    )

    # Save metrics in history to a separate file
    hv.save_history(history, filename="history_3d.csv")

    # Performance evaluation
    hv.visualize_history(history)

    print("Evaluating..")
    _, acc = model.evaluate(test_gen, verbose=1)
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

def split_dataset(dataset, dataset_size, split_ratio=0.2):
    # Ensure the split ratio is a float between 0 and 1
    if not (0 <= split_ratio <= 1):
        raise ValueError("split_ratio must be between 0 and 1.")
    
    # Calculate sizes for train and test datasets
    test_size = int(split_ratio * dataset_size)
    train_size = dataset_size - test_size

    # Handle the case where the dataset is too small
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Dataset size is too small for the specified split ratio.")

    # Create train and teimport unet3dst datasets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    
    return train_dataset, test_dataset


# Example usage:
input_dir = './preprocessed_data3d/input_data'
label_dir = './preprocessed_data3d/labels'
batch_size = 16

# Load and split dataset
train, validate, test = get_data_list(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)

# Define and train U-Net model
model = unet_3d.define_unet_3d()
model = train_unet(train, validate, test, model, batch_size=batch_size)
model.save("model_3d.h5")