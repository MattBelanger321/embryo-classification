import os
import tensorflow as tf
import cv2
import numpy as np

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

class UNetBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(dataset)
        self.iterator = iter(self.dataset)  # Create an iterator

    def __len__(self):
        # Calculate the number of batches
        return np.ceil(self.dataset_size / self.batch_size).astype(int)

    def __getitem__(self, idx):
        batch = next(self.iterator)

        # Inspect the batch here
        print(f"\nInspecting batch {idx}:\n")
        
        # Separate inputs and labels
        (input_data, output_data), (input_files, output_files) = batch

        # If you're returning filenames as part of the dataset, adjust as follows:
        # input_data, output_data, input_files = batch

        # Save the batch to disk
        for i in range(len(input_data)):
            # Assuming input_files contains file names
            filename = f"image_{i}"  # You could get the actual file name if included in the dataset
            create_directory_structure(f"./running_data/batch{idx}/")
            
            # Save input and label as images
            input_image = (input_data[i].numpy() * 255).astype(np.uint8)
            label_image = (output_data[i].numpy() * 255).astype(np.uint8)
            
            # Save images
            cv2.imwrite(f"./running_data/batch{idx}/{filename}_input.png", input_image)
            cv2.imwrite(f"./running_data/batch{idx}/{filename}_label.png", label_image)
        
        return input_data, output_data
    
    def on_epoch_end(self):
        """Resets the iterator at the end of each epoch."""
        self.iterator = iter(self.dataset)

