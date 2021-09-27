# model_kd: model in the paper
from resnet18 import resnet18, Conv_BN
from keras.layers import Input, ReLU, add, Lambda, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, \
                         GlobalAveragePooling2D, Dense, DepthwiseConv2D
from keras.losses import categorical_crossentropy
from keras.models import Model
import tensorflow as tf
import keras.backend as K
from loss import att_loss, ce_loss


def modelKD(input_shape=(512,512,3), n_classes=100, bifpn_repeats=1):

    inpt = Input(input_shape)
    gt = Input((n_classes,))

    # resnet back
    outs = resnet18(input_shape, width=64, n_classes=n_classes, notop=False)(inpt)
    s_feats = outs[:4]
    s_logits = outs[-1]     # logits, (b,cls)

    # bifpn
    t_feats = s_feats
    for i in range(bifpn_repeats):
        t_feats = bifpn(t_feats, i==0)

    # teacher head
    x = GlobalAveragePooling2D()(t_feats[-1])
    t_logits = Dense(n_classes, use_bias=True)(x)   # logits

    # loss
    loss_ce_s = Lambda(ce_loss, name='ce_s')([gt, s_logits])   # (b,1), keep batch axis for multi_gpu
    loss_ce_t = Lambda(ce_loss, name='ce_t')([gt, t_logits])
    loss_kd = Lambda(att_loss, name='kd_s')([s_logits, t_logits, *s_feats, *t_feats])
    loss_kd = Lambda(lambda args: tf.Print(args[0], [K.mean(i) for i in args],
                     message='loss_ce_s, loss_ce_t, loss_kd'))(
                     [loss_ce_s, loss_ce_t, loss_kd])
    loss = Lambda(add_loss)([loss_ce_s, loss_ce_t, loss_kd])

    model = Model([inpt, gt], loss)

    return model


def bifpn(feats, first_time, width=2, num_filters=[64,128,256,512]):
    # Node id starts from the input features and monotonically increase whenever
    # a new node is added. Here is an example for level P3 - P7:
    #     F4 (3)   L4 (3)              T4 (9)
    #     F3 (2)   L3 (2)    P3 (4)    T3 (8)
    #     F2 (1)   L2 (1)    P2 (5)    T2 (7)
    #     F1 (0)   L1 (0)              T1 (6)
    fpn_nodes = [{'node_idx': 4, 'feature_level': 4, 'src_nodes': [2,3],
                  'downsamp': False, 'upsamp': True},     # P3
                 {'node_idx': 5, 'feature_level': 3, 'src_nodes': [1,4],
                  'downsamp': False, 'upsamp': True},     # P2
                 {'node_idx': 6, 'feature_level': 2, 'src_nodes': [0,5],
                  'downsamp': False, 'upsamp': True},     # T1
                 {'node_idx': 7, 'feature_level': 3, 'src_nodes': [1,5,6],
                  'downsamp': True, 'upsamp': False},     # T2
                 {'node_idx': 8, 'feature_level': 4, 'src_nodes': [2,4,7],
                  'downsamp': True, 'upsamp': False},     # T3
                 {'node_idx': 9, 'feature_level': 5, 'src_nodes': [3,8],
                  'downsamp': True, 'upsamp': False},     # T4
    ]
    entry = feats
    if first_time:
        # 1x1 conv-bn-relu, wider
        feats = [Conv_BN(f,num_filters[i]*width, 1, strides=1, activation=True) for i,f in enumerate(feats)]

    for i, fpn_node in enumerate(fpn_nodes):
        nodes_in = []
        for idx, node_idx in enumerate(fpn_node['src_nodes']):
            x = feats[node_idx]
            n_filters = num_filters[fpn_node['feature_level']-2] * width
            # resample last node
            if fpn_node['downsamp'] and idx==len(fpn_node['src_nodes'])-1:
                # downsamp + 1x1 DCB
                x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
                x = DepthConvBlock(x, n_filters, kernel_size=1)
            elif fpn_node['upsamp'] and idx==len(fpn_node['src_nodes'])-1:
                # upsamp + 1x1 DCB
                x = UpSampling2D(size=2, interpolation='nearest')(x)
                x = DepthConvBlock(x, n_filters, kernel_size=1)
            else:
                # lateral: 3x3 DCB
                if i>2 and node_idx<4:
                    x = entry[node_idx]
                x = DepthConvBlock(x, n_filters, kernel_size=3)
            nodes_in.append(x)
        # weight sum
        new_node = Lambda(weighted_sum)(nodes_in)
        feats.append(new_node)

    return feats[-4:]


def DepthConvBlock(x, n_filters, kernel_size, strides=1, padding='same', repeats=1):
    if kernel_size==1:
        # conv-bn-relu
        x = Conv_BN(x, n_filters, kernel_size, strides, activation=True)
    else:
        # depthwise Conv
        x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        # pointwise Conv
        x = Conv2D(n_filters, kernel_size=1, strides=strides, padding=padding, use_bias=False)(x)
        # bn-relu
        x = ReLU()(BatchNormalization()(x))
        # repeats
        for i in range(repeats):
            x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=False)(x)
            x = Conv2D(n_filters, kernel_size=1, strides=strides, padding=padding, use_bias=False)(x)
            x = ReLU()(BatchNormalization()(x))
    return x


def weighted_sum(nodes_in):
    weights = [ReLU()(tf.Variable(1.)) for i in nodes_in]
    normed_weights = tf.unstack((tf.stack(weights)/(K.sum(weights)+K.epsilon())))
    new_node = add([nodes_in[i] * normed_weights[i] for i in range(len(nodes_in))])
    return new_node


def add_loss(args):
    loss_ce_s, loss_ce_t, loss_kd = args
    return loss_ce_s + loss_ce_t + loss_kd


if __name__ == '__main__':

    kd_model = modelKD()
    kd_model.summary()

    





