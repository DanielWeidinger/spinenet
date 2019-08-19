import matplotlib.pyplot as plt
import matplotlib.image as ilib
import tensorflow as tf
import numpy as np
plt.style.use('ggplot')

def get_image(path, before=False, after=False):
    img = ilib.imread(path)
    if before:
        plt.imshow(img)
        plt.show()
    
    img = ilib.imread(path)[..., :3]
    img = (img/127.5) - 1 #scale into [1, -1] range
    img = tf.image.resize(img, (160, 160))

    if after:
        plt.imshow(img)
        plt.show()

    img = tf.expand_dims(img, axis=0)

    return img

def plot_images(dataset, reps=5):

    x = dataset.take(5).repeat(reps)

    for output in x:
        plt.figure()
        plt.imshow(np.array(output[0]))
        plt.show()

    #for images in x:
    #    if(output[:, row*160:(row+1)*160].shape[1] == 0):
    #        break
    #    output[:, row*160:(row+1)*160] = np.vstack(images[0].numpy())
    #    print(row)
    #    row += 1
