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
	sample_counter = 1
	for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):

		matricies = []
		rgb_filepath = ""
		i = 0
		for (case_name, day, slice, class_name, matrix) in segmentation_mask.values():
			# Create the output path for each case and day
			case_output_path = os.path.join(output_directory, "plain_segments", case_name, f"{case_name}_{day}")
			rgb_output_path = os.path.join(output_directory, "rgb_overlays", case_name, f"{case_name}_{day}")
			create_directory_structure(case_output_path)

			# Generate the output image filename
			output_filename = f"{case_name}_{day}_{slice}_{class_name}.png"
			output_filepath = os.path.join(case_output_path, output_filename)

			rgb_filename = f"{case_name}_{day}_{slice}.png"
			rgb_filepath = os.path.join(rgb_output_path, rgb_filename)

			# Convert boolean matrix to uint8 image
			image = (matrix * 255).astype(np.uint8)  # Convert boolean to uint8
			
			# Save the image using OpenCV
			if sample_counter % 1000 == 0:
				print(f"writing file #{sample_counter}: {output_filepath}")
			sample_counter += 1
			i += 1
			matricies.append(image)
			cv2.imwrite(output_filepath, image)
		
		create_directory_structure(rgb_output_path)
		# read training image as cv_mat
		# multiplying by 5 to boost contrast
		original_image = cv2.cvtColor(cv2.imread(original_file, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR) * 10
		# Note: OpenCV uses BGR format
		rgb_segments = cv2.merge(matricies)	# give each segment a colour code (B=large bowel, G=small, R=stomach)
		mask = cv2.cvtColor(cv2.merge(matricies), cv2.COLOR_BGR2GRAY) > 0
		# Superimpose the RGB image only on non-black pixels
		# Use np.where to blend the images based on the mask
		result_image = np.where(mask[:, :, None], rgb_segments, original_image)
		cv2.imwrite(rgb_filepath, result_image)  
		cv2.imwrite(os.path.join(case_output_path,f"{case_name}_{day}_{slice}_original.png" ), original_image)  # Copy file preserving metadata

# Main application code
if __name__ == "__main__":
	save_segment_images()  # This will now save images one case at a time
