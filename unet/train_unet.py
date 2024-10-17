from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.optimizers import SGD

def define_unet():
    inputs = Input(shape=(512, 512, 1))  # Adjust input shape as needed

	# we use padding same so that the input doesnt change during convolutions
	# the naming convention is
    # c for convolution
    # d for down sampling
    # u for up sammpling
    # the numbers correspond to the position of the group in the U,
    # see section 2 of the UNet paper and figure 1

    # Encoder (downsampling)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c1)
    d1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c2)
    d2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c3)
    d3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c4)
    d4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck, this is the bottom of the U
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(d4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c5)

    # Decoder (upsampling)
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])  # Skip connection
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])  # Skip connection
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])  # Skip connection
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])  # Skip connection
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(c9)

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)  # Use softmax for multi-class segmentation

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
