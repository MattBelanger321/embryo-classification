import shutil
import cv2
import os
import numpy as np
import parse_training_csv as parser

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_segment_images(csv_file_path='./data/train.csv', output_directory='./generated_masks'):
	for case_name, day, slice, class_name, matrix, original_file in parser.parse_gi_tract_training_data(csv_file_path):
        # Create the output path for each case and day
		case_output_path = os.path.join(output_directory, case_name, f"{case_name}_{day}")
		create_directory_structure(case_output_path)

        # Generate the output image filename
		output_filename = f"{case_name}_{day}_{slice}_{class_name}.png"
		output_filepath = os.path.join(case_output_path, output_filename)

        # Convert boolean matrix to uint8 image
		image = (matrix * 255).astype(np.uint8)  # Convert boolean to uint8
        
        # Save the image using OpenCV
		print(f"writing {output_filepath}/{output_filename}")
		cv2.imwrite(output_filepath, image)
		shutil.copy2(original_file, os.path.join(case_output_path,f"{case_name}_{day}_{slice}_original.png" ))  # Copy file preserving metadata

# Main application code
if __name__ == "__main__":
	save_segment_images()  # This will now save images one case at a time
