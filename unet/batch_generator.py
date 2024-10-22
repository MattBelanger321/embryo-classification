import os
import tensorflow as tf
import cv2

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
        print(f"Files: {input_files}:")

        for i in range(len(input_files)):
            filename =  os.path.basename(input_files[i].numpy().decode("utf-8"))
            create_directory_structure(f"./running_data/batch{idx}/")
            cv2.imwrite(f"./running_data/batch{idx}/{filename}_input.png", input_data[i].numpy() * 255)
            cv2.imwrite(f"./running_data/batch{idx}/{filename}_labels.png", output_data[i].numpy() * 255)

        return input_data, output_data
