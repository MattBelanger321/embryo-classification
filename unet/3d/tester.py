from tensorflow.keras.models import load_model
import train_unet as tu
import numpy as np
from batch_generator import UNetBatchGenerator3D as batch_generator
from unet.metrics import metrics_calculator_3d as mc
import matplotlib.pyplot as plt

# TODO Change directory accordingly
input_dir = '../preprocessed_data3d/input_data'
label_dir = '../preprocessed_data3d/labels'
batch_size=16

def test3D(modelPath):
    # Load model
    model = load_model(modelPath)

    # Load and split dataset
    train_dataset, validate, test_dataset = tu.get_data_list(input_dir = input_dir, label_dir = label_dir, batch_size = batch_size)
    test_gen = batch_generator(test_dataset, batch_size)
    
    j=1
    plotting = {}
    for i in test_gen.iterator:
        y_pred = model.predict(i[0][0])
        metric = mc.accuracy_metric_3d(i[0][1], y_pred)
        plotting[j] = metric.numpy()
        print(metric.numpy())
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

test3D('../unet/3d/model_3d.h5')