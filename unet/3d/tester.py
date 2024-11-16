from tensorflow.keras.models import load_model
import train_unet as tu
import numpy as np
from unet.metrics import metrics_calculator_3d as mc

# TODO Change directory accordingly
input_dir = '/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/preprocessed_data3d/input_data'
label_dir = '/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/preprocessed_data3d/labels'
batch_size=16

def test3D(modelPath):
    # Load model
    model = load_model(modelPath)

    # Load and split dataset
    train_dataset, validate, test_dataset = tu.get_data_list(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)

    # Make predictions
    test, label = zip(*test_dataset)

    for j, k in zip(test, label):
        tensor = np.load(j)
        y_pred = model.predict(tensor.reshape(1, 5, 128, 128, 1))
        y_true = np.load(k)

        # Reduce from (1, 5, 128, 128, 3) to (5, 128, 128, 3)
        y_true = np.expand_dims(y_true, axis=0)

        print("Combined metric is: " + str(mc.accuracy_metric_3d(y_true, y_pred)))

test3D('/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/unet/3d/model_depth3_3d.h5')
