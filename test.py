from model import Spinenet
from dataloader import DataLoader, get_Idx
from util import plot_images
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from params import Parameter


#create model
params = Parameter().get_args()
sp = Spinenet(params)


dl = DataLoader(params)

data = dl.get_train()

model = Spinenet(params)
print(model.model.summary())
plot_images(data)