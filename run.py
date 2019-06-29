from model import Spinenet
from dataloader import DataLoader, get_poses
from util import get_image
import matplotlib.pyplot as plt
import argparse as args
import numpy as np

parser = args.ArgumentParser()

parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)

#dataloader
parser.add_argument('--data_path', default='./data/')
parser.add_argument('--save_path', default='./data/saved')
parser.add_argument('--img_size', default=160)
parser.add_argument('--val_percentage', default=5)

#training
parser.add_argument('--batch_size', default=32)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--shuffle_buffer_size', default=1000)
parser.add_argument('--model_path', default='models/model-v1.h5')

#model
parser.add_argument('--untrainable_layer', default=100)

#create model
params = parser.parse_args()
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

    res = sp.model.predict(img)
    print(res)