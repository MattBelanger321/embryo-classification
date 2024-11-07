from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD, Adam

def define_unet(width = 256, height = 256):
    inputs = Input(shape=(width, height, 1))  # Adjust input shape as needed

    # Encoder (downsampling)
    encoder_layers = []
    filters = 64
    x = inputs
    for _ in range(4):
        x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        encoder_layers.append(x)  # Save for skip connection
        x = MaxPooling2D((2, 2))(x)
        filters *= 2

    # Bottleneck
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)

    # Decoder (upsampling)
    for i in range(3, -1, -1):  # Loop backwards through encoder layers
        filters //= 2
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, encoder_layers[i]])  # Skip connection
        x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)


    outputs = Conv2D(3, (1, 1), activation='sigmoid')(x)  # Use softmax for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = Adam( clipnorm=1.0)  # Norm is clipped to 1.0
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
    
    return model