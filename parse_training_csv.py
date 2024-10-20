import numpy as np
import csv
import os
from pathlib import Path
from typing import Dict

# Assuming Classes is defined elsewhere
from gi_tract_classes import Classes

TRAINING_IMAGE_PATH = "./data/train/"

def find_file(directory, file_prefix):
    directory_path = Path(directory)
    # Use rglob to search for files that start with the given prefix
    for file in directory_path.rglob(f"{file_prefix}*"):
        if file.is_file():
            return file  # Return the first matching file's path
    return None  # Return None if no file is found

def get_dimensions(case, day, slice_name):
    directory = os.path.join(TRAINING_IMAGE_PATH, case, f"{case}_{day}/scans/")
    
    if not os.path.isdir(directory):
        raise Exception("Error: Training Data Not Found")
    
    png = find_file(directory, slice_name)
    tokenized_image_name = png.name.split("_")
    return int(tokenized_image_name[2]), int(tokenized_image_name[3]), png  # Ensure dimensions are integers

# Function to convert RLE to a boolean matrix
def rle_to_matrix(width, height, rle):
    bmatrix = np.zeros((height, width))  # Correctly initialize with (height, width)

    if len(rle) % 2 == 1:
        raise Exception("RLE Encoded image has odd length")

    for i in range(0, len(rle), 2):
        start = rle[i]
        run_length = rle[i + 1]

        # Convert 1D index to 2D coordinates
        row = start // width
        column = start % width

        for j in range(run_length):
            # Calculate new row and column
            new_row = row + (column + j) // width
            new_column = (column + j) % width

            # Ensure we don't go out of bounds
            if new_row < height:
                bmatrix[new_row][new_column] = 1
            else:
                raise Exception(f"Run length exceeds matrix bounds {new_row}x{new_column} {width}x{height} {i} {rle[0]}")

    return bmatrix

def parse_gi_tract_training_data(csv_file_path='./data/train.csv'):
	# Open the CSV file
	with open(csv_file_path, newline='') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)  # discard CSV header
        
		i = 1;
		all_segments = {}
		for row in reader:
			# Validate class
			if row[1] not in {Classes.STOMACH.value, Classes.LARGE_BOWEL.value, Classes.SMALL_BOWEL.value}:
				raise Exception(f"Class not recognized {row[:-1]}")

			tokens = row[0].split("_")
			width, height, file = get_dimensions(tokens[0], tokens[1], f"{tokens[2]}_{tokens[3]}")

			# Convert RLE string to a list of integers
			rle_values = []
			if len(row[2]) > 1:
				rle_values = list(map(int, row[2].split(" ")))
      
			mask = rle_to_matrix(width, height, rle_values)

			all_segments[row[1]] = (tokens[0], tokens[1], tokens[2]+tokens[3], row[1], mask)
			if i % 3 == 0:
				yield all_segments, file	# Yield case name, day, class, mask and file name for each of the 3 segments
				all_segments = {}	# reset
				i = 0
			i += 1