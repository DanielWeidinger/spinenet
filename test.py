from model import Spinenet
from dataloader import DataLoader, get_Idx
from util import get_image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from params import Parameter


#create model
params = Parameter().get_args()
sp = Spinenet(params)


dl = DataLoader(params)

batch_data = dl.get_train()
val_data = dl.get_val()
