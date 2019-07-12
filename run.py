from model import Spinenet
from dataloader import DataLoader, get_Idx
from util import get_image
from params import Parameter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


#create model
params = Parameter().get_args()
sp = Spinenet(params)

#training


if params.train:

    dl = DataLoader(params)

    batch_data = dl.get_train()
    val_data = dl.get_val()

    history = sp.model.fit(batch_data, epochs=20, validation_data=val_data)
    sp.model.save(params.model_path)

else:

    img = get_image('test_pics/dude_standing.jpg', before=False, after=False)

    res = sp.model.predict(img)[0]

    idx = np.argmax(res)
    print(f'{get_Idx()[idx]} with {res[idx] * 100}')

if params.convert:
    converter = tf.lite.TFLiteConverter.from_keras_model(sp.model)
    tflite_model = converter.convert()
    open(params.model_path + ".tflite", "wb").write(tflite_model)