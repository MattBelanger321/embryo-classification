from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import metrics_calculator_3d as mc

def define_unet_3d():
    inputs = Input(shape=(5, 128, 128, 1))  # Adjust the input shape if necessary
    # Encoder (downsampling)
    c1 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    c1 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c1)
    d1 = MaxPooling3D((1, 2, 2))(c1)

    c2 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d1)
    c2 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c2)
    d2 = MaxPooling3D((1, 2, 2))(c2)

    c3 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d2)
    c3 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c3)
    d3 = MaxPooling3D((1, 2, 2))(c3)

    c4 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d3)
    c4 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c4)
    d4 = MaxPooling3D((1, 2, 2))(c4)

    # Bottleneck
    c5 = Conv3D(1024, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d4)
    c5 = Conv3D(1024, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c5)

    # Decoder (upsampling)
    u6 = UpSampling3D((1, 2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u6)
    c6 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c6)

    u7 = UpSampling3D((1, 2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u7)
    c7 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c7)

    u8 = UpSampling3D((1, 2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u8)
    c8 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c8)

    u9 = UpSampling3D((1, 2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u9)
    c9 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c9)

    outputs = Conv3D(3, (1, 1, 1), activation='sigmoid')(c9)  # Adjust activation and filters for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = Adam(clipnorm=1.0)  # Norm is clipped to 1.0
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mc.dice_coefficient])
    
    return model
