from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD
import parse_training_csv as parser

from sklearn.model_selection import train_test_split

import numpy as np

import cv2

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def bce(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def load_data_and_test(model,csv_file_path='./data/train.csv', width=256, height=256):
    sample_counter = 1
    ind = 0
    # Parse the segmentation masks and original files
    preprocessed_labels = []
    input = [] # initalize an empty list to hold inputs
    use_sample = False
    for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):
        labels = []  # Initialize an empty list to hold labels
        for (case_name, day, slice_idx, class_name, matrix) in segmentation_mask.values():
            labels.append(matrix)  # Append the matrix to the labels list
            if True or case_name == "case30" and day == "day0" and slice_idx == "slice0097":
                use_sample = True

        # Convert the labels list to a NumPy array
        labels_array = np.asarray(labels)
        # Reshape labels_array to (height, width, channels) for OpenCV
        labels_array = np.transpose(labels_array, (1, 2, 0))
        labels_array = cv2.resize(labels_array, (width,height), interpolation=cv2.INTER_LINEAR)
        preprocessed_labels.append(labels_array)

        input_as_matrix = cv2.imread(original_file, cv2.IMREAD_GRAYSCALE)/255.0 # re-scale to (0,1)
        input_as_matrix = cv2.resize(input_as_matrix,(width,height), interpolation=cv2.INTER_LINEAR)
        input_as_matrix = np.asarray(input_as_matrix)
        input.append(input_as_matrix)

        if ind == 0 and use_sample:
            ind = len(input) - 1

        if sample_counter % 1000 == 0 or sample_counter == 38496:
            print(f"Searching #{sample_counter // 1000}, {use_sample}")
            # Evaluate the model on test data
            if use_sample:
                input = np.asarray(input).reshape(-1,256,256,1)
                labels = np.asarray(preprocessed_labels).reshape(-1,256,256,3)
                X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=1, random_state=42)
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                print(f"{test_loss},{test_accuracy}")

                sample_masks = model.predict(np.asarray(input[ind]).reshape(1,256,256,1))
                sample_masks = np.transpose(sample_masks, (3, 2, 1, 0))
                print(sample_masks.shape)
                cv2.imshow("h",sample_masks[0])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("h",sample_masks[1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("h",sample_masks[2])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("h", np.transpose(preprocessed_labels[ind], (2, 1, 0))[0])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("h", np.transpose(preprocessed_labels[ind], (2, 1, 0))[1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("h", np.transpose(preprocessed_labels[ind], (2, 1, 0))[2])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # return;
            preprocessed_labels = []  # reinitialize to an empty list to hold labels
            input = [] # reinitalize to an empty list to hold inputs
        
        sample_counter += 1

def load_data_and_test_3d(model, csv_file_path='./data/train.csv', width=128, height=128, depth=5):
    sample_counter = 1
    ind = 0
    # Parse the segmentation masks and original files
    preprocessed_labels = []
    input = []  # Initialize an empty list to hold inputs
    use_sample = False

    for segmentation_mask, original_file in parser.parse_gi_tract_training_data(csv_file_path):
        labels = []  # Initialize an empty list to hold labels for 3D

        for (case_name, day, slice_idx, class_name, matrix) in segmentation_mask.values():
            labels.append(matrix)
            if True or (case_name == "case123" and day == "day20" and slice_idx == "slice0097"):
                use_sample = True

        # Convert the labels list to a NumPy array for 3D volume
        labels_array = np.stack(labels, axis=0)  # Shape (depth, height, width)
        labels_array = np.transpose(labels_array, (1, 2, 0))  # Shape (height, width, depth)
        labels_array = cv2.resize(labels_array, (width, height), interpolation=cv2.INTER_LINEAR)
        preprocessed_labels.append(labels_array)

        # Load the single 3D input series
        img = cv2.imread(original_file, cv2.IMREAD_GRAYSCALE) / 255.0
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        input_series = np.repeat(img[np.newaxis, :, :], depth, axis=0)  # Replicate across depth if needed

        input.append(input_series)

        if ind == 0 and use_sample:
            ind = len(input) - 1

        if sample_counter % 1000 == 0 or sample_counter == 38496:
            print(f"Searching #{sample_counter // 1000}, {use_sample}")
            if use_sample:
                # Convert lists to arrays
                input = np.asarray(input).reshape(-1, depth, height, width, 1)
                labels = np.asarray(preprocessed_labels).reshape(-1, height, width, 3)
                
                # Split data for evaluation
                X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=1, random_state=42)
                # test_loss, test_accuracy = model.evaluate(X_test, y_test)
                # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

                # Predict on sample and visualize slices
                sample_prediction = model.predict(np.asarray(input[ind]).reshape(1, depth, height, width, 1))

                for i in range(depth):
                    cv2.imshow(f"Prediction Channel {i}", sample_prediction[0][i].reshape(height, width, 3))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    print(f"shape: {labels.shape}")
                    cv2.imshow(f"Ground Truth Channel {i}", labels[ind])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # return  # Exit after one sample is visualized

            preprocessed_labels = []
            input = []

        sample_counter += 1



# Load the pre-trained model
model = load_model('model.h5', custom_objects={'bce': bce})
load_data_and_test(model)

# Load the pre-trained model 3D
model_3d = load_model('model_3d.h5', custom_objects={'bce': bce})
load_data_and_test_3d(model_3d)