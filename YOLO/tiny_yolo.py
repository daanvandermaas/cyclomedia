#from keras.optimizers import SGD, Adam, RMSprop
#from keras.models import Model
#from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
#from keras.layers.advanced_activations import LeakyReLU
#
#
#anchors = [0.23,0.48, 0.31,0.62, 0.44,0.90, 0.60,1.25, 1.25,2.75]
#labels = list(["snelweg","voetganger","max80"])
#nb_box = len(anchors)//2
#nb_class = len(labels)
#max_box_per_image = 10
#
#input_image = Input(shape=(416,416, 3))
#true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))
#
## Layer 1
#x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
#x = BatchNormalization(name='norm_1')(x)
#x = LeakyReLU(alpha=0.1)(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)
#
## Layer 2 - 5
#for i in range(0,4):
#    x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
#    x = BatchNormalization(name='norm_' + str(i+2))(x)
#    x = LeakyReLU(alpha=0.1)(x)
#    x = MaxPooling2D(pool_size=(2, 2))(x)
#
## Layer 6
#x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
#x = BatchNormalization(name='norm_6')(x)
#x = LeakyReLU(alpha=0.1)(x)
#x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)
#
## Layer 7 - 8
#for i in range(0,2):
#    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
#    x = BatchNormalization(name='norm_' + str(i+7))(x)
#    x = LeakyReLU(alpha=0.1)(x)
#
#feature_extractor = Model(input_image, x)
#grid_h,grid_w = feature_extractor.get_output_shape_at(-1)[1:3]
#features = feature_extractor(input_image)
#
## make the object detection layer
#output = Conv2D(nb_box * (4 + 1 + nb_class), 
#                (1,1), strides=(1,1), 
#                padding='same', 
#                name='DetectionLayer', 
#                kernel_initializer='lecun_normal')(features)
#output = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)
#output = Lambda(lambda args: args[0])([output, true_boxes])
#
#model = Model([input_image, true_boxes], output)
#optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.save('uncompiled_tiny_yolo_2.h5')

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('uncompiled_tiny_yolo_2.h5')
model.load_weights('tiny_yolo_250_rot.h5')

model.save('tiny_yolo_voor_daan.h5')

dummy = np.zeros([1,1,1,1,10,4])

img = plt.imread('image.jpg')
img = np.expand_dims(img, axis=0)
# img moet de volgende shape hebben: 1,416,416,3

test = model.predict([img,dummy])
print(test)

