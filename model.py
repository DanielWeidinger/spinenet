import os
import numpy as np
import tensorflow as tf

class Spinenet():

    def __init__(self, params):

        if os.path.exists(params.model_path):
            print('Load existing model')
            self.model = tf.keras.models.load_model(params.model_path)
        else:
            IMG_SHAPE = (params.img_size, params.img_size, 3)

            base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
            

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:params.untrainable_layer]:
                layer.trainable =  False

            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            #TODO Try a second dense abstaction layer with Relu
            prediction_layer = tf.keras.layers.Dense(14, activation='softmax')

            self.model = tf.keras.Sequential([
                base_model,
                global_average_layer,
                prediction_layer
            ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=params.lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        
