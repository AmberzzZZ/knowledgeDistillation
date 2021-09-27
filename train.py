# from model_teacher import modelT
from model_kd import modelKD
from dataSequence import dataSequence
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
import keras.backend as K
import os
import sys


gpu_cnt = 1
if sys.platform=='linux':
    os.environ['CUDA_VISIVLE_DEVICES'] = '0,1'
    gpu_cnt = 2


def kdLrDecay(epoch):
    # decay by 10 when reach epoch [100, 150]
    lr_base = 0.1
    warmup_steps = 5
    if epoch + 1 < 5:
        # warmup
        return lr_base * (epoch+1) / warmup_steps
    elif epoch + 1 < 100:
        return lr_base
    elif epoch + 1 < 150:
        return lr_base * 0.1
    else:
        return lr_base * 0.01


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,single_model,multi_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = single_model
        self.multi_model = multi_model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        self.single_model.optimizer = self.multi_model.optimizer
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

    def on_epoch_end(self, epoch, logs=None):
        # save optimizer weights
        self.single_model.optimizer = self.multi_model.optimizer
        super(ParallelModelCheckpoint, self).on_epoch_end(epoch, logs)


if __name__ == '__main__':

    img_dir = "./data"
    log_dir = "./weights"

    num_classes = 100
    lr_linear = LearningRateScheduler(kdLrDecay, verbose=1)
    batch_size = 64 * gpu_cnt
    input_shape = (256,256)
    task_code = 'cifar'
    model_code = 'kd'

    # data
    # inputs: [(b,h,w,c), (b,cls)]
    # targets: (b,)
    train_generator = dataSequence(img_dir, input_shape, batch_size, num_classes,
                                   smoothing=0.05, mixup=True, rand_layers=2, rand_magnitude=10,
                                   pkl_dir='data/train.pkl', head='kd')
    test_generator = dataSequence(img_dir, input_shape, batch_size, num_classes,
                                  smoothing=0., mixup=False, rand_layers=0, rand_magnitude=10,
                                  pkl_dir='data/train.pkl', head='kd')

    # model
    optimizer = SGD(0.1, decay=0.0001)
    single_model = modelKD(input_shape+(3,), n_classes=num_classes, bifpn_repeats=1)
    single_model.compile(optimizer, loss=lambda y_true, y_pred: K.mean(y_pred))
    if gpu_cnt > 1:
        model = multi_gpu_model(single_model, gpu_cnt)
    else:
        model = single_model
    model.compile(optimizer, loss=lambda y_true, y_pred: y_pred)

    # ckpt
    filepath = log_dir + '/%s_input%d_cls%d_sgd_%s_epoch_{epoch:02d}_loss_{loss:.3f}.h5' % (model_code, input_shape[0], num_classes, task_code)

    if gpu_cnt>1:
        checkpoint = ParallelModelCheckpoint(single_model, model, filepath, monitor='loss', verbose=1, mode='auto', period=1)
    else:
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto')

    model.fit_generator(train_generator,
                        steps_per_epoch=10000//batch_size,
                        initial_epoch=0,
                        epoch=300,
                        validation_data=test_generator,
                        worker=16,
                        use_multiprocessing=True if sys.platform=='linux' else False,
                        callbacks=[checkpoint, lr_linear],
                        verbose=1)




    




