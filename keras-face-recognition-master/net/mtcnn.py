from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np 
import utils.utils as utils
import cv2

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

#Pnet粗略获取人脸框
#输出bbox位置和是否有人脸的置信度
#模型参考https://baijiahao.baidu.com/s?id=1644934945515389890&wfr=spider&for=pc
def create_Pnet(weight_path):
    #图像大小12*12*3
    input = Input(shape=[None, None, 3])    #创建3通道的模型输入

    #12*12*3->10*10*10
    x = Conv2D(10,        (3, 3),      strides=1, padding='valid',   name='conv1')(input) #用10个3*3的卷积核进行卷积，步长为1
              #卷积核个数  #卷积核大小  #步长       #卷积后不进行填充
    #Conv2D函数详解见https://blog.csdn.net/koala_cola/article/details/106883961
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)   #激活函数
    #10*10*10->5*5*10
    x = MaxPool2D(pool_size=2)(x)   #最大池化

    #5*5*10->3*3*16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x) #用16个3*3的卷积核进行卷积，步长为1
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)   #激活函数
    #3*3*16->1*1*32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x) #用32个3*3的卷积核进行卷积，步长为1
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)   #激活函数

    #1*1*32->1*1*2
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x) #用2个1*1的卷积核进行卷积 #无激活函数，线性。得到人脸二分类
    #1*1*32->1*1*4
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x) #用4个1*1的卷积核进行卷积，得到预测框位置

    model = Model([input], [classifier, bbox_regress])  #创建keras模型输入为input，输出为[classifier, bbox_regress]
    model.load_weights(weight_path, by_name=True)   #加载权重
    return model


#mtcnn的第二段Rnet
#精修框
def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])    #输入大小为24*24*3
    #24,24,3 -> 22,22,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input) #用28个3*3的卷积核进行卷积，步长为1
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x) #激活函数
    #22，22，28 -> 11,11,28
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x) #最大池化

    # 11,11,28 -> 9,9,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x) #用48个3*3的卷积核进行卷积，步长为1
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x) #激活函数
    #9,9,48->4,4,48
    x = MaxPool2D(pool_size=3, strides=2)(x)    #最大池化

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x) #用64个2*2的卷积核进行卷积，步长为1
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x) #激活函数
    # 3,3,64 -> 64,3,3并展平
    x = Permute((3, 2, 1))(x)   #将原来矩阵的3、3、64按3，2，1的顺序更换
    x = Flatten()(x)    #展平

    # 576 -> 128得到全连接层
    x = Dense(128, name='conv4')(x) #输入x*y大小的矩阵，输出x*128大小的矩阵
    x = PReLU( name='prelu4')(x)    #激活函数
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)  #人脸二分类，softmax为激活函数
    bbox_regress = Dense(4, name='conv5-2')(x)  #预测框的位置
    model = Model([input], [classifier, bbox_regress])  #创建keras模型输入为input，输出为[classifier, bbox_regress]
    model.load_weights(weight_path, by_name=True)   #加载权重
    return model


#mtcnn的第三段Onet
#精修框并获得人脸五个点
def create_Onet(weight_path):
    input = Input(shape = [48,48,3])
    # 48,48,3 -> 46,46,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input) #32个3*3卷积核
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    # 46,46,32->23,23,32
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)    #最大池化
    # 23,23,32 -> 21,21,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x) #64个3*3卷积核
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    # 21,21,64->10,10,64
    x = MaxPool2D(pool_size=3, strides=2)(x)    #最大池化
    # 10,10,64 -> 8,8,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x) #64个3*3卷积核
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    # 8,8,64 -> 4,4,64
    x = MaxPool2D(pool_size=2)(x)   #最大池化
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)    #128个2*2卷积核
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    # 3,3,128 -> 128,3,3
    x = Permute((3,2,1))(x) #维度置换
    x = Flatten()(x)    #展平
    # 1152 -> 256得到全连接层
    x = Dense(256, name='conv5') (x)    #输入x*y大小的矩阵，输出x*256大小的矩阵
    x = PReLU(name='prelu5')(x)

    # 鉴别
    # 256 -> 2 256 -> 4 256 -> 10 
    classifier = Dense(2, activation='softmax',name='conv6-1')(x)   #人脸二分类，softmax为激活函数
    bbox_regress = Dense(4,name='conv6-2')(x)   #预测框的位置
    landmark_regress = Dense(10,name='conv6-3')(x)  #人脸五个关键点位置

    model = Model([input], [classifier, bbox_regress, landmark_regress])    #创建keras模型输入为input，输出为[classifier, bbox_regress, landmark_regress]
    model.load_weights(weight_path, by_name=True)   #加载权重

    return model

class mtcnn():
    #初始化函数，加载三个网络模型
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')


    #使用MTCNN进行人脸检测
    def detectFace(self, img, threshold):   #img为输入图片
        #进行图像归一化
        copy_img = (img.copy() - 127.5) / 127.5 #归一化操作
        origin_h, origin_w, _ = copy_img.shape  #获得归一化后的图像高和宽

        #计算原始输入图像每一次缩放的比例
        scales = utils.calculateScales(img) #获得图像金字塔缩放比例列表
        out = []    #用来存放Pnet输出的列表

        #---------------------------------------------------------------#
        #Pnet网络粗略计算人脸框
        #将图像金字塔里的图片输入Pnet获得每一张图片人脸检测的初步特征提取效果
        for scale in scales:
            hs = int(origin_h * scale)  #将原始图像的高*缩放比例得到缩放后的高
            ws = int(origin_w * scale)  #将原始图像的宽*缩放比例得到缩放后的宽
            scale_img = cv2.resize(copy_img, (ws, hs))  #使用缩放后的宽和高进行缩放
            inputs = scale_img.reshape(1, *scale_img.shape) #第一个参数1表示将原图像通道数变为1，
                                                            #第二个参数表示矩阵行数
                                                            #这里的scale_img.shape返回的是一个表示图像矩阵的元组
                                                            #第一个参数为矩阵行数，第二个为矩阵列数，第三个为通道数
                                                            #这里*scale_img.shape表示reshape后矩阵的行数列数不变
            ouput = self.Pnet.predict(inputs)   #将reshape后的图像输入Pnet得到输出
            out.append(ouput)   #将输出结果添加到out列表

        image_num = len(scales) #图像金字塔图像数
        rectangles = [] #存放Pnet检测到的矩形框在原始图像中的矩形框位置的列表

        #遍历Pnet输出并解码得到原始图像中的矩形框位置
        for i in range(image_num):
            cls_prob = out[i][0][0][:,:,1]  #不知道这一步在干啥，有知道的写一下！！！好像是置信度
            roi = out[i][1][0]  #取出其对应的框的位置
            out_h, out_w = cls_prob.shape   #取出每个缩放后图片的高和宽
            out_side = max(out_h, out_w)    #取高和宽中较大者
            #解码过程
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)
            
        # 进行非极大抑制
        rectangles = utils.NMS(rectangles, 0.7)

        if len(rectangles) == 0:
            return rectangles

        #---------------------------------------------------------------#
        #Rnet部分稍微精确计算人脸框
        predict_24_batch = []   #用于保存Rnet输入的img列表
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]   #从原图中截取Pnet产生的矩形框
            scale_img = cv2.resize(crop_img, (24, 24))  #缩放到24*24，这是Rnet网络的输入要求
            predict_24_batch.append(scale_img)  #将缩放后的图像保存到Rnet输入列表里

        predict_24_batch = np.array(predict_24_batch)   #用输入列表初始化一个数组
        out = self.Rnet.predict(predict_24_batch)   #输入Rnet进行预测得到输出
        #图片中有没有人脸的可信度
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        #如何调整某一张图片对应的rectangle
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        #解码
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        #---------------------------------------------------------------#
        #Onet部分精确计算人脸框
        predict_batch = []  #用于保存Onet输入的img列表
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]   #从原图中截取Rnet产生的矩形框
            scale_img = cv2.resize(crop_img, (48, 48))  #缩放到48*48，这是Onet网络的输入要求
            predict_batch.append(scale_img)  #将缩放后的图像保存到Onet输入列表里

        predict_batch = np.array(predict_batch) #用输入列表初始化一个数组
        output = self.Onet.predict(predict_batch)   #输入Onet进行预测得到输出
        cls_prob = output[0]    #置信度
        roi_prob = output[1]    #调整方式
        pts_prob = output[2]    #人脸的五个特征点
        #筛选
        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles

