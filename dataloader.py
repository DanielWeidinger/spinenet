import tensorflow as tf
import numpy as np
import matplotlib.image as ilib
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
import cv2

class DataLoader():

    def __init__(self, params):
        self.params = params
        
        if not (os.path.exists(params.save_path)):
            self.preprocess()
        
        self.load_data()



    def preprocess(self):
        data = []
        label = []

        for path, _, files in os.walk(self.params.data_path):
            for file in files:
                ex = os.path.splitext(file)[1]
                if(ex != '.txt' and ex != '.sh'):
                    fpath = path + '/' + file

                    try:
                        pic = ilib.imread(fpath)#[..., :3]
                        if pic.shape[-1] == 4:
                            pic = cv2.cvtColor(pic, cv2.COLOR_BGRA2BGR)
                        pic = (pic/127.5) - 1 #scale into [1, -1] range
                        pic = tf.image.resize(pic, (self.params.img_size, self.params.img_size))
                        data.append(pic)
                        label.append(self.to_one_hot(path.split('/')[-1]))
                    except:
                        print("File corrupted: " + fpath)


        with open(self.params.save_path,"wb") as f:
            joblib.dump([data, label], f, protocol=2)

    def load_data(self):

        print('Start loading')

        with open(self.params.save_path,"rb") as f:
           [self.raw_data, self.raw_label] = joblib.load(f)

        dataset = tf.data.Dataset.from_tensor_slices((
            self.raw_data,
            self.raw_label
        )).shuffle(self.params.shuffle_buffer_size)

        all_data = len(self.raw_data)

        train_size = int(all_data * (100 - self.params.val_percentage)/100)
        val_size = all_data - train_size

        self.train = dataset.take(train_size)
        dataset.skip(train_size)

        self.val = dataset.take(val_size)

        print(f'Dataset Loaded. Features: {all_data}')
        print(f'Train {train_size}. Val: {val_size}')

    def get_train(self):
        return self.train.batch(self.params.batch_size)
    
    def get_val(self):
        return self.val.batch(self.params.batch_size)

    def to_one_hot(self, label):
        result = np.zeros(14)

        poses = get_poses()      

        result[poses.get(label)] = 1

        return result

def get_poses():
    return {
            'bridge' : 0,
            'camel' : 1,
            'chair' : 2,
            'chaturanga_dandasana' : 3,
            'cobra' : 4,
            'cow' : 5,
            'crescent_lunge' : 6,
            'half_moon' : 7,
            'plank' : 8,
            'tree' : 9,
            'triangle' : 10,
            'warrior_I' : 11,
            'warrior_II' : 12,
            'warrior_III' : 13
        } 

def get_Idx():
    return {
            0: 'bridge',
            1: 'camel',
            2: 'chair',
            3: 'chaturanga_dandasana',
            4: 'cobra',
            5: 'cow',
            6: 'crescent_lunge',
            7: 'half_moon',
            8: 'plank',
            9: 'tree',
            10: 'triangle',
            11: 'warrior_I',
            12: 'warrior_II',
            13: 'warrior_III'
        } 