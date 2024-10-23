import os
import tensorflow as tf
import cv2
import numpy as np

# Function to create directory structure
def create_directory_structure(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

        # print(f"{input_files.numpy().decode("utf-8")=}")
        #print(f"{input_data.numpy().min() = }, {input_data.numpy().max() = }, {input_data.numpy().shape = }, {input_data.numpy().dtype = } -  {output_data.numpy().min() = }, {output_data.numpy().max() = }, {output_data.numpy().shape = }, {output_data.numpy().dtype = }\n")

        # for i in range(input_data.numpy().shape[0]):
        #     input_filename = os.path.basename(input_files.numpy()[i].decode("utf-8")) # this is filthy
        #     label_filename = os.path.basename(output_files.numpy()[i].decode("utf-8"))
        #     input_path = f"./batch_images/batch_{idx}"
        #     label_path = f"./batch_images/batch_{idx}"
        #     create_directory_structure(input_path)
        #     create_directory_structure(label_path)
        #     cv2.imwrite(f"{input_path}/{input_filename}_input.png", (input_data * 255).astype(np.uint8))
        #     cv2.imwrite(f"{label_path}/{label_filename}_output.png", (output_data * 255).astype(np.uint8))

        return input_data, output_data
