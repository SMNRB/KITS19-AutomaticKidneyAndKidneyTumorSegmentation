IMAGE_ORDERING = 'channel_last'
ALPHA = 0.1

def yolo(no_classes, input=(512, 512, 3)):
    
    # Input layer
    input_image = Input(shape=input)
 
    # Layer 1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = layers.BatchNormalization(name='norm_1')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    
    # Layer 2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_2')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 3
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_3')(x)
    up_3 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 4 
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(up_3)
    x = layers.BatchNormalization(name='norm_4')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 5
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_5')(x)
    up_5 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(up_5)
    
    # Layer 6
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_6')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 7
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_7')(x)
    up_7 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 8
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(up_7)
    x = layers.BatchNormalization(name='norm_8')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Layer 9
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_9')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 10
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_10')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 11
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_11')(x)
    up_11 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 12
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(up_11)
    x = layers.BatchNormalization(name='norm_12')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 13
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_13')(x)
    up_13 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    x = layers.MaxPool2D(pool_size=(2, 2))(up_13)
    
    # Layer 14
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_14')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    dropout_14 = Dropout(rate=0.5, name='dropout_14')(x)
    
    # Layer 15
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(dropout_14)
    x = layers.BatchNormalization(name='norm_15')(x)
    up_15 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 16
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(up_15)
    x = layers.BatchNormalization(name='norm_16')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 17
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_17')(x)
    up_17 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 18
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(up_17)
    x = layers.BatchNormalization(name='norm_18')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 19
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_19')(x)
    up_19 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    dropout_19 = Dropout(rate=0.5, name='dropout_19')(x)
    
    # Layer 20
    layer_20 = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(dropout_19)
    x = layers.BatchNormalization(name='norm_20')(layer_20)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Layer 21
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_22')(x)
    up_21 = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Center
    x = layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_23')(up_21)
    
    # Upscale 21
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_22b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_21 = concatenate([x, up_21], axis=3, name='up_mergbe_9_2')
    
    # Upscale Layer 20
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20b', use_bias=False)(up_merge_21)
    x = layers.BatchNormalization(name='norm_20b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Upscale Layer 19
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_19b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_19 = concatenate([x, up_19], axis=3, name='up_merghe_9_2')
    
    # Upscale Layer 18
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18b', use_bias=False)(up_merge_19)
    x = layers.BatchNormalization(name='norm_18b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 17
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_17b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_17 = concatenate([x, up_17], axis=3, name='up_mergge_9_2')
    
    # Upscale Layer 16
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16b', use_bias=False)(up_merge_17)
    x = layers.BatchNormalization(name='norm_16b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)

    # Upscale Layer 15
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_15b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_15 = concatenate([x, up_15], axis=3, name='up_merges_9_2')
    
    # Upscale Layer 14
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14b', use_bias=False)(up_merge_15)
    x = layers.BatchNormalization(name='norm_14b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Upscale Layer 13
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_13b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_13 = concatenate([x, up_13], axis=3, name='up_mferge_9_2')
    x = UpSampling2D(size=(2, 2))(up_merge_13)
    
    # Upscale Layer 12
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_12b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 11
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_11b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Upscale Layer 10
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_10b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 9
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_9b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 8
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='conv_8b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_8b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 7
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_7b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    up_merge_7 = concatenate([x, up_7], axis=3, name='up_merge_9_d2')
    
    # Upscale Layer 6
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6b', use_bias=False)(up_merge_7)
    x = layers.BatchNormalization(name='norm_6b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Upscale Layer 5
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_5b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Upscale Layer 4
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4b', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_4b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 3
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_b3', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_3b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2b', use_bias=False)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = layers.BatchNormalization(name='norm_2b')(x)
    x = layers.advanced_activations.LeakyReLU(alpha=ALPHA)(x)
    
    # Upscale Layer 1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1b', use_bias=False)(x)
    
    conv_out = Convolution2D(3, no_classes, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    softmax = Activation("softmax", name='softmax')(conv_out)
    
    # Return output
    print(model.summary())
                                    
    return model
