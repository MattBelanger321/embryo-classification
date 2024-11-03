## this file will go through the pre-processed data, and help divide it so we can train the model properly

import glob
import random

import os
import re

import tensorflow as tf
import numpy as np

def split_filenames(paired_filenames, val_ratio=0.1, test_ratio=0.1):
	# Shuffle the samples/label pairs together
	random.seed(42)
	random.shuffle(paired_filenames)
	
	# Calculate the number of samples for each set
	total_size = len(paired_filenames)
	val_size = int(total_size * val_ratio)
	test_size = int(total_size * test_ratio)
	train_size = total_size - val_size - test_size  # The remaining samples for training

	# Split the shuffled paired_filenames into three lists
	train_pairs = paired_filenames[:train_size]
	val_pairs = paired_filenames[train_size:train_size + val_size]
	test_pairs = paired_filenames[train_size + val_size:]

	return (zip(*train_pairs), zip(*val_pairs), zip(*test_pairs))

def concatenate_directory_entries(input_directories, label_directories):
    # Define a sorting key that extracts the leading number from each filename
    def numeric_key(file_name):
        match = re.match(r"(\d+)", file_name)  # Match leading digits
        return int(match.group(1)) if match else float('inf')  # Use 'inf' if no number is found

    def get_files_from_directories(directories):
        file_set = []
        for directory in directories:
            if os.path.isdir(directory):
                for file_name in sorted(os.listdir(directory), key=numeric_key):
                    file_path = os.path.join(directory, file_name)
                    if os.path.isfile(file_path):  # Ensure it's a file
                        file_set.append(file_path)
        return file_set
    
    train_input_files = get_files_from_directories(input_directories)
    train_label_files = get_files_from_directories(label_directories)
    
    return train_input_files, train_label_files


def create_dataset_3d(val_ratio=0.1, test_ratio=0.1, input_dir = './preprocessed_data3d/input_data', label_dir = './preprocessed_data3d/labels', batch_size = 16):
	if val_ratio + test_ratio >= 1:
		raise Exception("split exceeds 100%")

	# Get list of all input and label file directories
	# input_filenames[i] is the directory with all the patches for that case
	input_filenames = glob.glob(f"{input_dir}/*")
	label_filenames = glob.glob(f"{label_dir}/*")

	# Pair the filenames together
	paired_filenames = list(zip(input_filenames, label_filenames))

	# get the split
	train, validate, test = split_filenames(paired_filenames, val_ratio, test_ratio)
	(train_input_directories, train_label_directories) = train
	(validate_input_directories, validate_labels_directories) = validate
	(test_input_directories, test_labels_directories) = test

	# get list of training data from directory lists 
	(train_input, train_label) = concatenate_directory_entries(train_input_directories, train_label_directories)
	(validate_input, validate_label) = concatenate_directory_entries(validate_input_directories, validate_labels_directories)
	(test_input, test_label) = concatenate_directory_entries(test_input_directories, test_labels_directories)
     
	# Create a dataset from file paths
	train_dataset = tf.data.Dataset.from_tensor_slices((list(train_input), list(train_label)))
	validate_dataset = tf.data.Dataset.from_tensor_slices((list(validate_input), list(validate_label)))
	test_dataset = tf.data.Dataset.from_tensor_slices((list(test_input), list(test_label)))
    
    # Define a function to load and preprocess each pair of input/label npy files
	def process_npy_file(input_file, label_file):
        # Load the .npy files using numpy_function and cast them to float32
		input_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[input_file], Tout=tf.float32)
		label_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[label_file], Tout=tf.float32)
        
        # Explicitly set the shapes to ensure TensorFlow knows what to expect
		input_data.set_shape([5, 128, 128, 1])  # Assuming input is 256x256 grayscale images depth 5
		label_data.set_shape([5, 128, 128, 3])  # Assuming label is 256x256 with 3 classes depth 5

		return input_data, label_data

    # Shuffle dataset (you can adjust the buffer size based on your total data)
    # dataset = dataset.shuffle(buffer_size=len(input_filenames), seed=10)

    # Map the file-loading function to the dataset
	train_dataset_files = train_dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
	validate_dataset_files = validate_dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
	test_dataset_files = test_dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # add filenames for the generator for debugging
    # (file, filename)
	train_dataset = tf.data.Dataset.zip((train_dataset_files, train_dataset))
	validate_dataset = tf.data.Dataset.zip((validate_dataset_files, validate_dataset))
	test_dataset = tf.data.Dataset.zip((test_dataset_files, test_dataset))
     
	# Batch the dataset
	train_dataset = train_dataset.batch(batch_size)
	validate_dataset = train_dataset.batch(validate_dataset)
	test_dataset = train_dataset.batch(test_dataset)
     
	return train_dataset, validate_dataset, test_dataset

# Example usage:
input_dir = './preprocessed_data3d/input_data'
label_dir = './preprocessed_data3d/labels'
batch_size = 16

create_dataset_3d(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)