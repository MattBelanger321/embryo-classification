from tensorflow.keras.models import load_model
import train_unet as tu
import numpy as np
from unet.metrics import metrics_calculator as mc
from batch_generator import UNetBatchGenerator as batch_generator
import matplotlib.pyplot as plt

# TODO Update the path accordingly
input_dir = 'input_dir'
label_dir = 'label_dir'
batch_size=16

def test2DModel(modelPath):
    # Load model
    model = load_model(modelPath)

    # Load and split dataset
    train_dataset, validate, test_dataset = tu.get_data_list(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)
    test_gen = batch_generator(test_dataset, batch_size)
    
    j=1
    plotting = {}
    for i in test_gen.iterator:
        y_pred = model.predict(i[0][0])
        metric = mc.combined_metric(i[0][1], y_pred)
        plotting[j] = metric.numpy()
        j=j+1

    # Extract batch numbers (keys) and metric values (values)
    batch_numbers = list(plotting.keys())
    metrics_values = list(plotting.values())

    # Plot the values
    plt.figure(figsize=(8, 6))
    plt.plot(batch_numbers, metrics_values, marker='o', color='b', label='Metric')

    # Adding labels and title
    plt.xlabel('Batch Number')
    plt.ylabel('Metric Value')
    plt.title('Metric Value vs. Batch Number')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

# TODO Change path accordingly since we are not committing the models
test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/2d_with_dropout.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_2d_droupout_without_clip.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_depth3_2d.h5')
# test2DModel('/Users/rishabhtyagi/Desktop/Neural/gi-tract-classification/unet/2d/model_L2_2d.h5')