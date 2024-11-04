import os
import tensorflow as tf
import cv2
import numpy as np
import random

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

class UNetBatchGenerator3D(tf.keras.utils.Sequence):
    def __init__(self, paired_filenames, batch_size):
        self.original_filename_pairs = list(paired_filenames)  # Store the original dataset
        self.batch_size = batch_size
        random.seed(42)
        self.on_epoch_end()  # Initialize by shuffling and batching the dataset

    def __len__(self):
        # Calculate the number of batches
        return len(self.batched_dataset)

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
    
    def create_dataset(self):
        print("Created Dataset")
        input, labels = zip(*self.shuffled_filenames)

        # Create a dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices((list(input), list(labels)))
        
        # Define a function to load and preprocess each pair of input/label npy files
        def process_npy_file(input_file, label_file):
            # Load the .npy files using numpy_function and cast them to float32
            input_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[input_file], Tout=tf.float32)
            label_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[label_file], Tout=tf.float32)
            
            # Explicitly set the shapes to ensure TensorFlow knows what to expect
            input_data.set_shape([5, 128, 128, 1])  # Assuming input is 256x256 grayscale images depth 5
            label_data.set_shape([5, 128, 128, 3])  # Assuming label is 256x256 with 3 classes depth 5

            return input_data, label_data

        # Shuffle dataset (you can adjust the buffer size based on your total data)
        # dataset = dataset.shuffle(buffer_size=len(input_filenames), seed=10)

        # Map the file-loading function to the dataset
        dataset_files = dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
        
        # add filenames for the generator for debugging
        # (file, filename)
        dataset = tf.data.Dataset.zip((dataset_files, dataset))
        print("finished creating dataset")

        return dataset
    
    def on_epoch_end(self):
        print("Shuffling and rebatching dataset at the end of the epoch.")
        
        # Shuffle the filename pairs
        self.shuffled_filenames = self.original_filename_pairs[:]
        random.shuffle(self.shuffled_filenames)
        
        # Create a new dataset with the shuffled filenames and rebatch it
        self.batched_dataset = self.create_dataset().batch(self.batch_size)

        """Resets the iterator at the end of each epoch."""
        self.iterator = iter(self.batched_dataset)
        
        print("Finished Suffling")
