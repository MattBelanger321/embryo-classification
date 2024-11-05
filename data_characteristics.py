import csv

import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def undersample_3d(csv_file_path='./data/train.csv'):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # discard CSV header
        
        i = 0;
        slice = []
        empty_slices = 0 # this is the number of slices that contain none of the 3 classes
        empty_samples = 0 # this is the number of sample volumes that contain none of the 3 classes
        slice_count = 0  # number of slices seen
        sample_count = 0    # number of samples seen
        sample_has_classes = False  # set if the current sample has a classification
        for row in reader:
            slice.append(row)

            # Check if all 3 classes for the slice have been read
            if len(slice) != 3:
                 continue
            
            slice_count+=1
            if slice[0][2] == "" and slice[1][2] == "" and slice[2][2] == "":
                 empty_slices += 1
            else:
                sample_has_classes = True
            
            # if all 144 slices have been processed
            if slice_count % 144 == 0:
                sample_count += 1
                if not sample_has_classes:
                     empty_samples += 1
                sample_has_classes = False

            slice = []

        logging.info(f"{empty_slices}/{slice_count} empty slices and {empty_samples}/{sample_count} empty samples")

# Main application code
if __name__ == "__main__":
	undersample_3d()