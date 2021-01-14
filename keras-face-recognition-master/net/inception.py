from functools import partial
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add
from keras import backend as K


def scaling(x, scale):
    return x * scale
#生成卷积层的名字
def _generate_layer_name(name, branch_idx=None, prefix=None):
    #prefix前缀为None返回None
    if prefix is None:
        return None
    #branch_idx分支编号为None返回prefix_name
    if branch_idx is None:
        return '_'.join((prefix, name))
    #返回完整卷积层名，由三部分组成prefix_Branch_idx_name
    return '_'.join((prefix, 'Branch', str(branch_idx), name))

#与BN层合并的2维卷积层
def conv2d_bn(x,filters,kernel_size,strides=1,padding='same',activation='relu',use_bias=False,name=None):
    # 根据参数创建二维卷积层
    x = Conv2D(filters, #过滤器的个数
               kernel_size, #卷积核的数量
               strides=strides, #步长
               padding=padding, #“SAME”表示边界要填充，即给边界加上padding让卷积的输入和输出保持同样（SAME）尺寸；“VALID”表示边界不填充
               use_bias=use_bias,   #是否使用偏置层（BN）
               name=name)(x)
    # 如果use_bias 为 True,追加BN层 ,默认都在激活函数前添加BN层
    if not use_bias:
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                               scale=False, name=_generate_layer_name('BatchNorm', prefix=name))(x)
    # 激活函数
    if activation is not None:
        x = Activation(activation, name=_generate_layer_name('Activation', prefix=name))(x)
    return x

#Inception-ResNetV1中的Inception-ResNet-A，B，C部分的共用函数，通过block_type进行区分
def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 3
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
        
    name_fmt = partial(_generate_layer_name, prefix=prefix)#固定_generate_layer_name函数的prefix为prefix，生成name_fmt函数

    if block_type == 'Block35':     #Inception-ResNet-A部分
        # 分支1:一次1*1卷积
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1x1', 0))
        # 分支2:一次1*1卷积和一次3*3卷积
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 32, 3, name=name_fmt('Conv2d_0b_3x3', 1))
        # 分支3:一次1*1卷积和两次次3*3卷积
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0a_1x1', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0b_3x3', 2))
        branch_2 = conv2d_bn(branch_2, 32, 3, name=name_fmt('Conv2d_0c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]   #组织在一个列表中
    elif block_type == 'Block17':   #Inception-ResNet-B部分
        # 分支1:一次1*1卷积
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1x1', 0))
        # 分支2:一次1*1卷积,一次1*7卷积,一次7*1卷积
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 128, [1, 7], name=name_fmt('Conv2d_0b_1x7', 1))
        branch_1 = conv2d_bn(branch_1, 128, [7, 1], name=name_fmt('Conv2d_0c_7x1', 1))
        branches = [branch_0, branch_1] #组织在一个列表中
    elif block_type == 'Block8':    #Inception-ResNet-C部分
        # 分支1:一次1*1卷积
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1x1', 0))
        # 分支2:一次1*1卷积,一次1*3卷积,一次3*1卷积
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
        branch_1 = conv2d_bn(branch_1, 192, [1, 3], name=name_fmt('Conv2d_0b_1x3', 1))
        branch_1 = conv2d_bn(branch_1, 192, [3, 1], name=name_fmt('Conv2d_0c_3x1', 1))
        branches = [branch_0, branch_1] #组织在一个列表中

    mixed = Concatenate(axis=channel_axis, name=name_fmt('Concatenate'))(branches)  #利用列表的拼接完成卷积层堆叠
    up = conv2d_bn(mixed,K.int_shape(x)[channel_axis],1,activation=None,use_bias=True,
                   name=name_fmt('Conv2d_1x1')) #设置1x1的卷积处理（通道调整）
    up = Lambda(scaling,
                output_shape=K.int_shape(up)[1:],
                arguments={'scale': scale})(up)
    x = add([x, up])    #与未经处理的分支部分进行相加
    if activation is not None:  #激活函数
        x = Activation(activation, name=name_fmt('Activation'))(x)
    return x

#关于InceptionV1的详解参考博客https://blog.csdn.net/julialove102123/article/details/79632721
def InceptionResNetV1(input_shape=(160, 160, 3),    #输入图像大小160*160*3
                      classes=128,  #128种类别
                      dropout_keep_prob=0.8):
    channel_axis = 3
    inputs = Input(shape=input_shape)

    #Inception-ResNetV1的stem部分
    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')  #设置步长为2的，3x3的卷积处理   160,160,32->79,79,32
    x = conv2d_bn(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')  #3x3的卷积处理,卷积核的数量为32   79,79,32->78,78,32
    x = conv2d_bn(x, 64, 3, name='Conv2d_2b_3x3')   #3x3的卷积处理,卷积核的数量为64，78,78,32->77,77,32
    #经历了一次步长为2 x 2的最大池化，边长变为1/2，图片尺寸由77 x 77变成了38 x 38
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = conv2d_bn(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')  #设置1x1的卷积处理,卷积核的数量为80    38,38,64->37,37,80
    x = conv2d_bn(x, 192, 3, padding='valid', name='Conv2d_4a_3x3') #设置3x3的卷积处理,卷积核的数量为192   37,37,80->36,36,192
    x = conv2d_bn(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')  #设置步长为2的3x3的卷积处理,卷积核的数量为256   36,36,192->17,17,256

    #5次Inception-ResNet-A 处理
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,scale=0.17,block_type='Block35',block_idx=block_idx)

    # Reduction-A 部分:   17,17,256 -> 8,8,896
    name_fmt = partial(_generate_layer_name, prefix='Mixed_6a') #固定_generate_layer_name函数的prefix为Mixed_6a，生成name_fmt函数
    # 分支1：一次步长为2的384通道3x3的卷积
    branch_0 = conv2d_bn(x, 384, 3,strides=2,padding='valid',name=name_fmt('Conv2d_1a_3x3', 0)) #卷积层名：Mixed_6a_0_Conv2d_1a_3x3
    # 分支2：一次192通道1x1的卷积，一次192通道3x3的卷积，一次256通道3x3的卷积
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))  #卷积层名：Mixed_6a_1_Conv2d_0a_1x1
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))   #卷积层名：Mixed_6a_1_Conv2d_0b_3x3
    branch_1 = conv2d_bn(branch_1,256,3,strides=2,padding='valid',name=name_fmt('Conv2d_1a_3x3', 1))    #卷积层名：Mixed_6a_1_Conv2d_1a_3x3
    # 分支3：一次步长为2的最大池化
    branch_pool = MaxPooling2D(3,strides=2,padding='valid',name=name_fmt('MaxPool_1a_3x3', 2))(x)
    #三个部分的卷积层堆叠
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    #10次Inception-ResNet-B 处理
    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    # Reduction-B部分 8,8,896 -> 3,3,1792
    name_fmt = partial(_generate_layer_name, prefix='Mixed_7a')#固定_generate_layer_name函数的prefix为Mixed_7a，生成name_fmt函数
    # 分支1：一次256通道1x1的卷积，一次步长为2的384通道3x3的卷积
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))  #卷积层名：Mixed_7a_0_Conv2d_0a_1x1
    branch_0 = conv2d_bn(branch_0,384,3,strides=2,padding='valid',name=name_fmt('Conv2d_1a_3x3', 0))    #卷积层名：Mixed_7a_0_Conv2d_1a_3x3
    # 分支2：一次256通道1x1的卷积，一次步长为2的384通道3x3的卷积
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))  #卷积层名：Mixed_7a_1_Conv2d_0a_1x1
    branch_1 = conv2d_bn(branch_1,256,3,strides=2,padding='valid',name=name_fmt('Conv2d_1a_3x3', 1))    #卷积层名：Mixed_7a_1_Conv2d_1a_3x3
    # 分支3：一次256通道1x1的卷积，一次256通道3x3的卷积，一次步长为2的256通道3x3的卷积
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))  #卷积层名：Mixed_7a_2_Conv2d_0a_1x1
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))   #卷积层名：Mixed_7a_2_Conv2d_0b_3x3
    branch_2 = conv2d_bn(branch_2,256,3,strides=2,padding='valid',name=name_fmt('Conv2d_1a_3x3', 2))    #卷积层名：Mixed_7a_2_Conv2d_1a_3x3
    # 分支4：一次步长为2的最大池化
    branch_pool = MaxPooling2D(3,strides=2,padding='valid',name=name_fmt('MaxPool_1a_3x3', 3))(x)   #卷积层名：Mixed_7a_3_MaxPool_1a_3x3
    #三个部分的卷积层堆叠
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    #5次Inception-ResNet-C 处理
    for block_idx in range(1, 6):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,scale=1.,activation=None,block_type='Block8',block_idx=6)

    # 平均池化
    x = GlobalAveragePooling2D(name='AvgPool')(x) #平均池化层
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)#防止在训练中过拟合
    # 全连接层到128维度
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = _generate_layer_name('BatchNorm', prefix='Bottleneck')#得到全连接层名Bottleneck_BatchNorm
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
                           name=bn_name)(x)#追加BN层

    # 创建模型
    model = Model(inputs, x, name='inception_resnet_v1')

    return model
