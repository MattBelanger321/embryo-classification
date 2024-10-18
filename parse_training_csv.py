# This file should parse the "./data/train.csv" file
# and be able to return the information from each of the 3 segments
# in some standardized manner
# you should seralize the data into a boolean matrix segment[i,j]
# such that segment[i,j] = 0 if the pixel at coordinate (i,j) is not in the class
# and is 1 otherwise
# since there are 3 classes for each slice, we should have 3 matricies for each slice. Most of these matricies will be blank
# I recommened using opencv

from xml.etree.ElementPath import find
import numpy as np
import csv
import os
from pathlib import Path

from gi_tract_classes import Classes

# Helper function to read kaggle data

TRAINING_IMAGE_PATH = "./data/train/"

def find_file(directory, file_prefix):
    directory_path = Path(directory)
    # Use rglob to search for files that start with the given prefix
    for file in directory_path.rglob(f"{file_prefix}*"):  # Match any file that starts with the prefix
        if file.is_file():  # Check if it's a file
            return str(file.name)  # Return the first matching file's path
    return None  # Return None if no file is found

def get_dimensions(case,day,slice):
	directory = f"{TRAINING_IMAGE_PATH}{case}/{case}_{day}/scans/"

	if not os.path.isdir(directory):
		raise Exception("Error: Training Data Not Found")
	
	png = find_file(directory,slice)
	tokenized_image_name = png.split("_")
	return tokenized_image_name[2],tokenized_image_name[3]
	

def parse_gi_tract_training_data(csv_file_path = './data/train.csv'):
	# Open the CSV file
	with open(csv_file_path, newline='') as csvfile:
		reader = csv.reader(csvfile)
		next(reader) # discard csv header
		
		for row in reader:
			tokens = row[0].split("_")
			(width,height) = get_dimensions(tokens[0], tokens[1], tokens[2] + "_" + tokens[3])
			print(f"{width}x{height}")
			return

        