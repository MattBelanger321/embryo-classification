## this file will go through the pre-processed data, and help divide it so we can train the model properly

import glob
import random

import os

def count_cases(input_dir):
	return 267

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
    def get_files_from_directories(directories):
        file_set = set()  # Use a set to avoid duplicates
        for directory in directories:
            if os.path.isdir(directory):
                for file_name in os.listdir(directory):
                    file_path = os.path.join(directory, file_name)
                    if os.path.isfile(file_path):  # Ensure it's a file
                        file_set.add(file_path)
        return list(file_set)
    
    train_input_files = get_files_from_directories(input_directories)
    train_label_files = get_files_from_directories(label_directories)
    
    return train_input_files, train_label_files


def create_dataset(val_ratio=0.1, test_ratio=0.1, input_dir = './preprocessed_data3d/input_data', label_dir = './preprocessed_data3d/labels', batch_size = 16):
	if val_ratio + test_ratio >= 1:
		raise Exception("split exceeds 100%")
	
	# number of distinct samples
	case_count = count_cases(input_dir)

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





# Example usage:
input_dir = './preprocessed_data3d/input_data'
label_dir = './preprocessed_data3d/labels'
batch_size = 16

create_dataset(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)