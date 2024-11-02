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
        self.batch_count = len(dataset)
        self.iterator = iter(self.dataset)  # Create an iterator

    def __len__(self):
        # Calculate the number of batches
        return self.batch_count

    def __getitem__(self, idx):
        try:
            # TODO: investiagte dataset.skip function to simulate random access
            batch = next(self.iterator)
        except StopIteration:
            # Tensorflow does pre-fetches at the beggining of datasets
            # we need to use iterators because we are using a take dataset
            # tensorflow will not reset our iterator
            # this means we are gaurenteed to run out of bounds
            # so we need to loop back around when we do
            self.iterator = iter(self.dataset)
            batch = next(self.iterator)

        # Inspect the batch here
        print(f"\nInspecting batch {idx}:\n")
        
        # Separate inputs and labels
        (input_data, output_data), (input_files, output_files) = batch

        for i in range(len(input_files)):
            filename = os.path.basename(input_files[i].numpy().decode("utf-8"))
            patch_input = input_data[i].numpy()  # Extract the 3D patch for input
            patch_output = output_data[i].numpy()  # Extract the 3D patch for labels

            # Ensure the directory for this batch exists
            create_directory_structure(f"./running_data3d/batch{idx}/{filename}")

            # Loop through each slice in the depth dimension of the 3D patch
            for z in range(patch_input.shape[0]):  # Assuming depth is the first dimension
                input_slice = patch_input[z, :, :] * 255  # Scale slice to [0, 255] for visualization
                label_slice = patch_output[z, :, :] * 255  # Scale slice to [0, 255] for visualization
                
                # Save each slice as an image
                cv2.imwrite(f"./running_data3d/batch{idx}/{filename}/slice{z+1}_input.png", input_slice.astype(np.uint8))
                cv2.imwrite(f"./running_data3d/batch{idx}/{filename}/slice{z+1}_labels.png", label_slice.astype(np.uint8))
        
        return input_data, output_data
    
    def on_epoch_end(self):
        print("ENDING")
        """Resets the iterator at the end of each epoch."""
        self.iterator = iter(self.dataset)

