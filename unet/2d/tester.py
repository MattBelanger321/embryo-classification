from tensorflow.keras.models import load_model
import train_unet as tu
import numpy as np
from unet.metrics import metrics_calculator as mc

# TODO Update the path accordingly
input_dir = '/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/preprocessed_data2d/input_data'
label_dir = '/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/preprocessed_data2d/labels'
batch_size=16

def test2DModel(modelPath):
    # Load model
    model = load_model(modelPath)

    # Load and split dataset
    train_dataset, validate, test_dataset = tu.get_data_list(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)

    # Make predictions
    test, label = zip(*test_dataset)

    for j, k in zip(test, label):
        tensor = np.load(j)
        y_pred = model.predict(tensor.reshape(1, 128, 128, 1))
        y_true = np.load(k)

        # Reduce from (1, 128, 128, 3) to (128, 128, 3)
        y_pred = np.squeeze(y_pred)

        print("Combined metric is: " + str(mc.combined_metric(y_true, y_pred)))

# TODO Change path accordingly since we are not committing the models
test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/2d_with_dropout.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_2d_droupout_without_clip.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_depth3_2d.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_L2_2d.h5')