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

def load_and_test_3d(model, csv_file_path='./data/train.csv', output_directory='./preprocessed_data3d', width=128, height=128, depth=5, stride=3):
    sample_counter = 1
    input = []  # List to hold input patches
    preprocessed_labels = []  # List to hold label patches
    ind = 0  # Index for a specific patch if needed
    use_sample = False  # Flag to select a specific test sample

    logging.info("Starting 3D data loading and testing")

    for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):
        labels = []

        for (case_name, day, slice_idx, class_name, matrix) in segmentation_mask.values():
            labels.append(matrix)
            if case_name == "case30" and day == "day0" and slice_idx == "slice0097":
                use_sample = True

        labels_array = np.asarray(labels).reshape(-1, width, height, 3)
        input_as_matrix = np.stack(
            [cv2.imread(f, cv2.IMREAD_GRAYSCALE) / 255.0 for f in original_file], axis=0
        ).reshape(-1, width, height, 1)

        num_patches = (input_as_matrix.shape[0] - depth) // stride + 1
        for j in range(num_patches):
            start = j * stride
            input_patch = input_as_matrix[start:start + depth]
            label_patch = labels_array[start:start + depth]

            input.append(input_patch)
            preprocessed_labels.append(label_patch)

            if j == 0 and use_sample:
                ind = len(input) - 1

        if sample_counter % 1000 == 0 or sample_counter == 38496:
            logging.info(f"Testing batch {sample_counter // 1000}")
            if use_sample:
                input_np = np.asarray(input)
                labels_np = np.asarray(preprocessed_labels)

                X_train, X_test, y_train, y_test = train_test_split(input_np, labels_np, test_size=1, random_state=42)
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

                sample_masks = model.predict(input_np[ind].reshape(1, depth, width, height, 1))
                sample_masks = np.transpose(sample_masks, (3, 2, 1, 0))

                for channel in range(sample_masks.shape[0]):
                    cv2.imshow(f"Predicted Channel {channel}", sample_masks[channel])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                for channel in range(preprocessed_labels[ind].shape[-1]):
                    cv2.imshow(f"True Label Channel {channel}", np.transpose(preprocessed_labels[ind], (3, 2, 1, 0))[channel])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                return

            input, preprocessed_labels = [], []  # Reset for the next batch
        sample_counter += 1

    logging.info("3D data loading and testing complete.")


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
