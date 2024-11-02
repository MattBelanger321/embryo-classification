import shutil
import cv2
import os
import numpy as np
import parse_training_csv as parser


import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_segment_images(csv_file_path='./data/train.csv', output_directory='./generated_masks'):
	sample_counter = 1
	for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):

		segment_matricies = []
		rgb_filepath = ""
		i = 0
		for (case_name, day, slice, class_name, matrix) in segmentation_mask.values():
			# Create the output path for each case and day
			case_output_path = os.path.join(output_directory, "plain_segments", case_name, f"{case_name}_{day}")
			rgb_output_path = os.path.join(output_directory, "rgb_overlays", case_name, f"{case_name}_{day}")
			create_directory_structure(case_output_path)
			create_directory_structure(rgb_output_path)

			# Generate the output image filename
			output_filename = f"{case_name}_{day}_{slice}_{class_name}.png"
			output_filepath = os.path.join(case_output_path, output_filename)

			rgb_filename = f"{case_name}_{day}_{slice}.png"
			rgb_filepath = os.path.join(rgb_output_path, rgb_filename)

			# Convert boolean matrix to uint8 image
			segment_mask = (matrix * 255).astype(np.uint8)  # Convert boolean to uint8

			# Save the image using OpenCV
			if sample_counter % 1000 == 0:
				print(f"writing file #{sample_counter}: {output_filepath}")
			sample_counter += 1
			i += 1
			segment_matricies.append(segment_mask)
			cv2.imwrite(output_filepath, segment_mask)
		
		create_directory_structure(rgb_output_path)
		# read training image as cv_mat
		original_image = parser.normalize(cv2.imread(original_file, cv2.IMREAD_GRAYSCALE)) * 255
		original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
		# Note: OpenCV uses BGR format
		rgb_segments = cv2.merge(segment_matricies)	# give each segment a colour code (B=large bowel, G=small, R=stomach)
		mask = cv2.cvtColor(rgb_segments, cv2.COLOR_BGR2GRAY) > 0
		# Superimpose the RGB image only on non-black pixels
		# Use np.where to blend the images based on the mask
		result_image = np.where(mask[:, :, None], rgb_segments, original_image)
		cv2.imwrite(rgb_filepath, result_image)  
		cv2.imwrite(os.path.join(case_output_path,f"{case_name}_{day}_{slice}_original.png" ), original_image)  # Copy file preserving metadata

def save_preprocessed_training_data2d(csv_file_path='./data/train.csv', output_directory='./preprocessed_data2d'):
	input_dir = f"{output_directory}/input_data"
	label_dir = f"{output_directory}/labels"
	create_directory_structure(input_dir)
	create_directory_structure(label_dir)
	i = 1	# using a numeric_prefix for simplicity
	for input, labels, sample_id in parser.load_data(csv_file_path):
		input = np.asarray(input).reshape(-1,256,256,1)
		labels = np.asarray(labels).reshape(-1,256,256,3)
		np.save(f"{input_dir}/{i}_{sample_id}.npy", input.astype(np.float32))
		np.save(f"{label_dir}/{i}_{sample_id}.npy", labels.astype(np.float32))
		i += 1

def save_preprocessed_training_data3d(csv_file_path='./data/train.csv', output_directory='./preprocessed_data3d', depth=5, stride=3):
    input_dir = f"{output_directory}/input_data"
    label_dir = f"{output_directory}/labels"
    i = 1  # using a numeric_prefix for simplicity
    logging.info("Starting 3D data preprocessing")
    logging.info(f"CSV File Path: {csv_file_path}")
    logging.info(f"Output Directory: {output_directory}")
    logging.info(f"Patch Depth: {depth}, Stride: {stride}")
    for input, labels, sample_id in parser.load_data3d(csv_file_path, width=128, height=128):
        sample_input_dir = os.path.join(input_dir,f"{i}_{sample_id}")
        sample_label_dir = os.path.join(label_dir,f"{i}_{sample_id}")
        # Ensure input and label dimensions are as expected
        input = np.asarray(input).reshape(-1, 128, 128, 1)
        labels = np.asarray(labels).reshape(-1, 128, 128, 3)
        
        # Calculate the total number of patches given the specified depth and stride
        num_patches = (input.shape[0] - depth) // stride + 1
        logging.debug(f"input shape: {input.shape}")
        logging.info(f"Processing Sample ID: {sample_id} | Total Patches: {num_patches}")
        for j in range(num_patches):
            create_directory_structure(sample_input_dir)
            create_directory_structure(sample_label_dir)            

            # Create patches with the specified depth, moving with the specified stride
            start = j * stride
            input_patch = input[start:start+depth, :, :, :]
            label_patch = labels[start:start+depth, :, :, :]
            # Save each patch with the specified depth as a separate .npy file
            input_file_path = f"{sample_input_dir}/{j+1}_patch.npy"
            label_file_path = f"{sample_label_dir}/{j+1}_patch.npy"
            np.save(input_file_path, input_patch.astype(np.float32))
            np.save(label_file_path, label_patch.astype(np.float32))
            # Log each patch's file path after saving
            logging.debug(f"Saved input patch to: {input_file_path}")
            logging.debug(f"Saved label patch to: {label_file_path}")
        i += 1

    logging.info("3D data preprocessing complete.")

def display_labels_3d(csv_file_path='./data/train.csv', width=256, height=256, num_slices=144):
    for volume_images, volume_labels, slice_id in parser.load_data3d(csv_file_path, width, height, num_slices):
        print(f"Displaying volume for ID: {slice_id}")
        
        # Iterate through each slice in the volume
        for i in range(num_slices):
            # Retrieve the label for the current slice
            label_slice = volume_labels[i]  # Shape (height, width, channels)

            # Convert the label to a displayable RGB image
            if label_slice.shape[-1] == 1:  # If single-channel, convert to 3 channels
                label_rgb = cv2.cvtColor(label_slice, cv2.COLOR_GRAY2RGB)
            else:
                label_rgb = cv2.normalize(label_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Display the label slice as an RGB image
            cv2.imshow(f"Label Slice {i+1} of {slice_id}", label_rgb)
            cv2.waitKey(100)  # Display for 100 ms
            
            # Close all windows after each slice display
            if cv2.waitKey(0) & 0xFF == ord('q'):
                # Close all windows after displaying the entire volume
                cv2.destroyAllWindows()
                break

# Main application code
if __name__ == "__main__":
	# save_segment_images()  # This will now save images one case at a time
	# save_preprocessed_training_data2d()
	save_preprocessed_training_data3d()	## lot
