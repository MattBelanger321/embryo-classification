from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from unet.metrics import metrics_calculator_3d as mc

def define_unet_3d(patch_size = 5, width = 128, height = 128):
    inputs = Input(shape=(patch_size, width, height, 1))  # Adjust the input shape if necessary

    # Encoder (downsampling)
    encoder_layers = []
    filters = 64
    x = inputs
    for i in range(4):
        x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        encoder_layers.append(x)  # Save for skip connection
        x = MaxPooling3D((1, 2, 2))(x)
        filters *= 2

    # Bottleneck
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)

    # Decoder (upsampling)
    for i in range(3, -1, -1):  # Loop backwards through encoder layers
        filters //= 2
        x = UpSampling3D((1, 2, 2))(x)
        x = concatenate([x, encoder_layers[i]])  # Skip connection
        x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)

    # Output layer
    outputs = Conv3D(3, (1, 1, 1), activation='sigmoid')(x)  # Adjust activation and filters for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = Adam(clipnorm=1.0)  # Norm is clipped to 1.0
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mc.accuracy_metric_3d])
    
    return model
