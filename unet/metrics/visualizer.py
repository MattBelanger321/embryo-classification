import pickle
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import importlib.util
import metrics_calculator as mc
import metrics_calculator_3d as mc3

# TODO Use relative path
directory_2d = '/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/unet/2d'
directory_3d = '/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/unet/3d'

# Plot Loss
# TODO Code optimization: Pass models and pickles as list to function so that we can iterate over that list
def visualizeLossAndAccuracy():
    # Load object from pickle file
    # 2D BASE UNET
    history_file_2d_base = open(os.path.join(directory_2d, "history_full2D.pickle"),'rb')
    history_object_2d_base = pickle.load(history_file_2d_base)

    # Load 2D UNET WITH DROPOUT for getting custom_metric
    # TODO Use relative path
    withDropout_2D = load_model('/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/unet/2d/2d_with_dropout.h5', custom_objects={'combined_metric': mc.combined_metric})

    # 2D UNET WITH DROPOUT
    history_file_2d_dropout = open(os.path.join(directory_2d, "history_full2D_dropout.pickle"),'rb')
    history_object_2d_dropout = pickle.load(history_file_2d_dropout)

    # 2D UNET WITH DROPOUT WITH CLIP NORM
    history_file_2d_dropout_clip_norm = open(os.path.join(directory_2d, "history_full2D_dropout_clip_norm.pickle"),'rb')
    history_object_2d_dropout_clip_norm = pickle.load(history_file_2d_dropout_clip_norm)

    # 2D UNET L2 REGULARIZATION
    history_file_2d_l2_reg = open(os.path.join(directory_2d, "history_full2D_L2.pickle"),'rb')
    history_object_2d_l2_reg = pickle.load(history_file_2d_l2_reg)

    # 2D UNET DEPTH 3
    history_file_2d_depth3 = open(os.path.join(directory_2d, "history_full2D_depth3.pickle"),'rb')
    history_object_2d_depth3 = pickle.load(history_file_2d_depth3)
    print(history_object_2d_depth3.history.keys())

    # 3D BASE UNET
    history_file_3d_base = open(os.path.join(directory_3d, "history_full3D.pickle"),'rb')
    history_object_3d_base = pickle.load(history_file_3d_base)

    plt.figure(figsize=(12, 4))

    # Training loss vs validation loss
    # This plot for loss has 1 row, 2 columns and first among the 2 we are going to plot
    plt.subplot(1, 2, 1)
    plt.plot(history_object_2d_base.history['loss'], label='Loss - 2d base unet')
    plt.plot(history_object_2d_dropout.history['loss'], label='Loss - 2d unet with dropout')
    plt.plot(history_object_2d_dropout_clip_norm.history['loss'], label='Loss - 2d unet with dropout clip norm')
    plt.plot(history_object_2d_l2_reg.history['loss'], label='Loss - 2d unet L2 regularization')
    plt.plot(history_object_2d_depth3.history['loss'], label='Loss - 2d unet Depth 3')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.ylim(0, 0.1)
    plt.legend()

    # Plot combined metric
    plt.subplot(1, 2, 2)
    plt.plot(history_object_2d_dropout.history['combined_metric'], label='Combined Metric - 2d unet with dropout')
    plt.plot(history_object_2d_dropout_clip_norm.history['combined_metric'], label='Combined Metric - 2d unet with dropout and clip norm')
    plt.plot(history_object_2d_l2_reg.history['combined_metric'], label='Combined Metric - 2d unet L2 regularization')
    plt.plot(history_object_2d_depth3.history['combined_metric'], label='Combined Metric - 2d unet Depth 3')
    plt.title('combined_metric over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Combined Metric')
    plt.ylim(0, 0.4)
    plt.legend()

    plt.figure(figsize=(12, 8))
    # Plot Accuracy
    plt.subplot(1, 1, 1)
    plt.plot(history_object_2d_base.history['accuracy'], label='Accuracy - 2d base unet')
    plt.plot(history_object_3d_base.history['accuracy'], label='Accuracy - 3d base unet')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

    # 3D Model Graphs
    # Load 3D UNET WITH DROPOUT for getting custom_metric
    withDropout_3D = load_model('/Users/meshwa/Desktop/COMP 8610/gi-tract-classification/unet/3d/model_3d_dropout.h5', custom_objects={'accuracy_metric_3d': mc3.accuracy_metric_3d})

    # 3D UNET WITH DROPOUT WITH CLIP NORM
    history_file_3d_dropout_clip_norm = open(os.path.join(directory_3d, "history_full3D_dropout_with_clip.pickle"),'rb')
    history_object_3d_dropout_clip_norm = pickle.load(history_file_3d_dropout_clip_norm)

    # 3D UNET WITH DROPOUT WITHOUT CLIP NORM
    history_file_3d_dropout_without_clip_norm = open(os.path.join(directory_3d, "history_full3D_dropout_without_clip.pickle"),'rb')
    history_object_3d_dropout_without_clip_norm = pickle.load(history_file_3d_dropout_without_clip_norm)

    # 3D UNET DEPTH 3
    history_file_3d_depth3 = open(os.path.join(directory_3d, "history_full3D_depth3.pickle"),'rb')
    history_object_3d_depth3 = pickle.load(history_file_3d_depth3)

    # 3D UNET L2 NORM
    history_file_3d_l2 = open(os.path.join(directory_3d, "history_full3D_l2_regularization.pickle"),'rb')
    history_object_3d_l2 = pickle.load(history_file_3d_l2)

    plt.figure(figsize=(12, 4))

    # Training loss vs validation loss
    # This plot for loss has 1 row, 2 columns and first among the 2 we are going to plot
    plt.subplot(1, 2, 1)
    plt.plot(history_object_3d_base.history['loss'], label='Loss - 3D base unet')
    plt.plot(history_object_3d_dropout_clip_norm.history['loss'], label='Loss - 3D unet dropout with clip normalization')
    plt.plot(history_object_3d_dropout_without_clip_norm.history['loss'], label='Loss - 3D unet dropout without clip normalization')
    plt.plot(history_object_3d_depth3.history['loss'], label='Loss - 3D unet depth 3')
    plt.plot(history_object_3d_l2.history['loss'], label='Loss - 3D L2 Norm')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.ylim(0, 0.075)
    plt.legend()

    # Plot combined metric
    plt.subplot(1, 2, 2)
    plt.plot(history_object_3d_dropout_clip_norm.history['accuracy_metric_3d'], label='Combined Metric - 3d unet with dropout')
    plt.plot(history_object_3d_dropout_without_clip_norm.history['accuracy_metric_3d'], label='Combined Metric - 3d unet with dropout without clip norm')
    plt.plot(history_object_3d_depth3.history['accuracy_metric_3d'], label='Combined Metric - Depth 3')
    plt.plot(history_object_3d_l2.history['accuracy_metric_3d'], label='Combined Metric - 3d unet L2 regularization')
    plt.title('Combined Metric over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Combined Metric')
    plt.legend()

    # Plot combined metric
    plt.tight_layout()
    plt.show()

visualizeLossAndAccuracy()