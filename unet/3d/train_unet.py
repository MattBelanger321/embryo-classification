import unet3d

def train_unet(train, test, model, batch_size=32, epochs=2, spe=2, vsteps=1, save_path="model_epoch_{epoch:02d}.keras"):
    print("Fitting...")
    # Fit model and validate on test data after each epoch

    train_gen = batch_generator(train, batch_size) # the training data generator
    test_gen =  batch_generator(test, batch_size) # the training data generator

    print(f"{train}")
    print(f"{test}")
    print(f"{test_gen.__len__()}")

    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=save_path,   # Path for saving model; {epoch:02d} allows you to save by epoch number
        save_weights_only=False,  # Set to True if you want to save only weights
        save_freq='epoch',        # Save after each epoch
        verbose=1
    )

    # Pass the checkpoint callback to the fit function
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        verbose=1,
        steps_per_epoch=spe,
        validation_steps=vsteps,
        callbacks=[checkpoint_callback]
    )

    print("Evaluating..")
    _, acc = model.evaluate(test_gen, verbose=1)
    print('Test Accuracy: %.3f' % (acc * 100.0))
     
    return model

# Function to check for NaN values in a batch of input/label pairs
def contains_nan(batch):
    inputs, labels = batch
    # Check if any NaN values are present in the inputs or labels separately
    nan_in_inputs = tf.reduce_any(tf.math.is_nan(inputs))
    nan_in_labels = tf.reduce_any(tf.math.is_nan(labels))
    return nan_in_inputs or nan_in_labels


# Function to check the entire dataset for NaN values
def dataset_has_nan(dataset):
    for batch in dataset:
        if contains_nan(batch):
            print("NaN values found in the dataset!")
            return True
    print("No NaN values found in the dataset.")
    return False


def is_float32(dataset):
    for batch in dataset:
        # Assuming the batch is of the form (inputs, labels)
        inputs, labels = batch
        
        # Check if dtype of inputs and labels is tf.float32
        if inputs.dtype != tf.float32 or labels.dtype != tf.float32:
            print("Data is not Float32")
            return False
            
    print("Data is float32")
    return True

def is_normalized(dataset):
    for batch in dataset:
        # Assuming the dataset is of the form (inputs, labels)
        inputs, labels = batch
        
        # Check if all values in inputs are within the range [0, 1]
        if not tf.reduce_all((inputs >= 0) & (inputs <= 1)):
            print("Inputs are not on [0,1]")
            return False
        
        # Check if all values in labels are within the range [0, 1]
        if not tf.reduce_all((labels >= 0) & (labels <= 1)):
            print("Labels are not on [0,1]")
            return False
            
    print("Data is normalized")
    return True


# This function will check if the dataset is fit for training
def validate_dataset(dataset):
    return not dataset_has_nan(dataset) and is_float32(dataset) and is_normalized(dataset)

def get_dataset(input_dir, label_dir, batch_size):
    # Get list of all input and label files
    input_filenames = glob.glob(f"{input_dir}/*.npy")
    label_filenames = glob.glob(f"{label_dir}/*.npy")

    # Pair the filenames together
    paired_filenames = list(zip(input_filenames, label_filenames))

    # Shuffle the pairs together
    random.seed(42)
    random.shuffle(paired_filenames)

    # Unzip the shuffled pairs back into separate lists
    input_filenames, label_filenames = zip(*paired_filenames)

    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((list(input_filenames), list(label_filenames)))
    
    # Define a function to load and preprocess each pair of input/label npy files
    def process_npy_file(input_file, label_file):
        # Load the .npy files using numpy_function and cast them to float32
        input_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[input_file], Tout=tf.float32)
        label_data = tf.numpy_function(func=lambda f: np.load(f).astype(np.float32), inp=[label_file], Tout=tf.float32)
        
        # Remove any singleton dimensions (e.g., [1, 256, 256, 1] -> [256, 256, 1])
        input_data = tf.squeeze(input_data, axis=0)  # Remove the unnecessary first dimension
        label_data = tf.squeeze(label_data, axis=0)  # Remove the unnecessary first dimension
        
        # Explicitly set the shapes to ensure TensorFlow knows what to expect
        input_data.set_shape([256, 256, 1])  # Assuming input is 256x256 grayscale images
        label_data.set_shape([256, 256, 3])  # Assuming label is 256x256 with 3 classes

        return input_data, label_data

    # Shuffle dataset (you can adjust the buffer size based on your total data)
    # dataset = dataset.shuffle(buffer_size=len(input_filenames), seed=10)

    # Map the file-loading function to the dataset
    dataset_files = dataset.map(process_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
    
    # add filenames for the generator for debugging
    # (file, filename)
    dataset = tf.data.Dataset.zip((dataset_files, dataset))

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # if not validate_dataset(dataset):
    #     raise Exception("Data is INVALID")

    return dataset, len(input_filenames) // batch_size

def split_dataset(dataset, dataset_size, split_ratio=0.2):
    # Ensure the split ratio is a float between 0 and 1
    if not (0 <= split_ratio <= 1):
        raise ValueError("split_ratio must be between 0 and 1.")
    
    # Calculate sizes for train and test datasets
    test_size = int(split_ratio * dataset_size)
    train_size = dataset_size - test_size

    # Handle the case where the dataset is too small
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Dataset size is too small for the specified split ratio.")

    # Create train and teimport unet3dst datasets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    
    return train_dataset, test_dataset


# Usage
flushable_stream = FlushableStream.FlushableStream("output.log", flush_interval=2)  # Flush every 2 seconds
sys.stdout = flushable_stream  # Redirect stdout

# Example usage:
input_dir = './preprocessed_data2d/input_data'
label_dir = './preprocessed_data2d/labels'
batch_size = 32

# Load and split dataset
dataset, batch_count = get_dataset(input_dir, label_dir, batch_size)

train_dataset, test_dataset = split_dataset(dataset, batch_count)
# Define and train U-Net model
model = unet3d.define_unet()
model = train_unet(train_dataset, test_dataset, model, batch_size=32)
model.save("model.h5")