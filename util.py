import matplotlib.pyplot as plt
import matplotlib.image as ilib
import tensorflow as tf
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
    