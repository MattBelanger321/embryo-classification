import cv2
import os
import numpy as np
import parse_training_csv as parser

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_segment_images(training_data, output_directory='./generated_images'):
    for case, segments in training_data.items():
        # Create the output path for each case
        case_output_path = os.path.join(output_directory, case)
        create_directory_structure(case_output_path)
        
        for class_name, matrix in segments.items():
            # Generate the output image filename
            output_filename = f"{case}_{class_name}.png"
            output_filepath = os.path.join(case_output_path, output_filename)

            # Convert boolean matrix to uint8 image
            image = (matrix * 255).astype(np.uint8)  # Convert boolean to uint8
            
            # Save the image using OpenCV
            cv2.imwrite(output_filepath, image)

# Main application code
training_data = parser.parse_gi_tract_training_data()
save_segment_images(training_data)
