import os
import tensorflow as tf
import cv2

class UNetBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = len(dataset)
        
		# Create an iterator
        self.iterator = iter(self.dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Fetch batch
        batch = next(self.iterator)  # Get the next batch        
        # Inspect the batch here
        print(f"\nInspecting batch {idx}:\n")
        
        # Separate inputs and labels
        (input_data, output_data), (input_files, output_files) = batch

        first_input_filename = os.path.basename(os.path.abspath(input_files[0].numpy().decode("utf-8")))
        first_label_filename = os.path.basename(os.path.abspath(output_files[0].numpy().decode("utf-8")))
        print(first_input_filename)
        print(first_label_filename)

        cv_input = input_data.numpy()[0].transpose([2,0,1])
        cv2.imwrite(f"{first_input_filename}.png",cv_input[0]*255)

        exit()

        return input_data, output_data
