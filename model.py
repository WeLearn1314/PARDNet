from keras.models import Model
from keras.layers import Input, Add,Subtract, PReLU, Conv2DTranspose, \
    Concatenate, MaxPooling2D, UpSampling2D, Dropout, concatenate, GlobalAveragePooling2D,\
    Reshape, Dense, multiply, Activation
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow.compat.v1 as tf
import os
import math
tf.disable_v2_behavior()
tf.enable_eager_execution()

class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

# Training 100H
def PSNR(y_true, y_pred):
    y_true_t = y_true*255.0
    max_pixel = 255.0
    y_pred_t = K.clip(y_pred*255.0, 0.0, 255.0)
    # y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred_t - y_true_t))))


def get_model(model_name="the_end"):

    if model_name == "the_end":
        return get_the_end_model()
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


def get_the_end_model(input_channel_num=3, feature_dim=64, resunit_num=16):

    def _back_net(inputs):
        def _residual_block(inputs,number):
            x = Conv2D(feature_dim, (3, 3), dilation_rate=(1,1), padding="same", kernel_initializer="he_normal")(inputs)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_dim, (3, 3), dilation_rate=(2,2), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_dim, (3, 3), dilation_rate=(3,3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            
            channels = K.int_shape(x)[-1]
            # print(channels)
            t = int(abs((math.log(channels,2)+1)/2))
            k = t if t%2 else t+1
            x_global_avg_pool = GlobalAveragePooling2D()(x)
            eca = Reshape((channels,1))(x_global_avg_pool)
            eca = Conv1D(1,kernel_size=k,padding="same")(eca)
            eca = Activation('sigmoid')(eca)
            eca = Reshape((1, 1, channels))(eca)
            x = multiply([x, eca])
            
            m = Add()([x, inputs])
            return m


        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = PReLU(shared_axes=[1, 2])(x)
        x0 = x

        for i in range(resunit_num):
            x = _residual_block(x, 4)

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x0])
        x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        return x

    def _rain_net(inputs):
        def _residual_block(inputs, number):
            x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            
            channels = K.int_shape(x)[-1]
            t = int(abs((math.log(channels,2)+1)/2))
            k = t if t%2 else t+1
            x_global_avg_pool = GlobalAveragePooling2D()(x)
            eca = Reshape((channels,1))(x_global_avg_pool)
            eca = Conv1D(1,kernel_size=k,padding="same")(eca)
            eca = Activation('sigmoid')(eca)
            eca = Reshape((1, 1, channels))(eca)
            x = multiply([x, eca])
            
            m = Add()([x, inputs])
            return m

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
        x = PReLU(shared_axes=[1, 2])(x)
        x0 = x

        for i in range(resunit_num):
            x = _residual_block(x, 4)

        x = Conv2D(feature_dim, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([x, x0])
        x = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        return x

    inputs = Input(shape=(None, None, input_channel_num), name='Rain_image')
    Rain0 = _rain_net(inputs)
    Rain1 = _back_net(inputs)

    Rain = Concatenate(axis=-1)([Rain0,Rain1])
    Rain = Conv2D(input_channel_num, (3, 3), padding="same", kernel_initializer="he_normal")(Rain)
    
    out = Subtract()([inputs, Rain])

    model = Model(inputs=inputs, outputs=[out])
    return model


def main():
    # model = get_model()
    model = get_model("unet")
    # model.summary()


if __name__ == '__main__':
    main()
