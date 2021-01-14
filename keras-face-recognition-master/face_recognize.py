import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1

class face_rec():
    def __init__(self):
        self.mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
        self.threshold = [0.5,0.8,0.9]  #门限函数                        作用还不知道

        #载入facenet将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        # model.summary()
        model_path = './model_data/facenet_keras.h5'    #facenet模型文件路径
        self.facenet_model.load_weights(model_path)     #载入facenet权值

        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset")  #获取face_dataset数据库文件夹下所有文件

        self.known_face_encodings=[]    #存储数据库的编码后的人脸(人脸特征向量)
        self.known_face_names=[]    #存储数据库图片的人名
        #依次对数据库中数据提取人脸特征向量
        for face in face_list:
            name = face.split(".")[0]   #获得文件名作为人名

            img = cv2.imread("./face_dataset/"+face)    #读取图片
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   #从BGR转到RGB


            rectangles = self.mtcnn_model.detectFace(img, self.threshold)  # 利用facenet_model检测人脸

            # 转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片  一张图片只有一个人脸，所以直接取下标0
            rectangle = rectangles[0]
            # 记下他们的landmark
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            #截取人脸部分图像
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))
            #人脸对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            #扩展一个维度
            new_img = np.expand_dims(new_img,0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

            self.known_face_encodings.append(face_encoding) #放入known_face_encodings列表
            self.known_face_names.append(name)#放入known_face_names列表

    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw) #获得图片长宽
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB) #从BGR转到RGB

        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)  # 利用facenet_model检测人脸

        # 没检测到人脸
        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        #限制人脸框的位置在整张图片之内
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = [] #保存所有人脸特征向量
        face_names = []  # 保存检测结果

        #依次对图片中的每张人脸提取特征向量
        for rectangle in rectangles:
            # 记下landmark
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 截取人脸部分图像
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))
            # 人脸对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            # 扩展一个维度
            new_img = np.expand_dims(new_img,0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encodings.append(face_encoding)    #添加到face_encodings列表

        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分，要求距离小于0.9则识别成功
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.9)
            name = "Unknown"    #默认为Unknown
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的下标
            best_match_index = np.argmin(face_distances)
            # 人脸与数据库中匹配上，将name改为数据库中对应人名
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name) #添加到face_names列表

        rectangles = rectangles[:,0:4] #取出人脸位置
        #-----------------------------------------------#
        #   画框~!~
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) #显示人名
        return draw

if __name__ == "__main__":

    dududu = face_rec() #创建人脸识别类对象
    #此处被注释代码不要删除
    # video_capture = cv2.VideoCapture(0)

    # while True:
    #     ret, draw = video_capture.read()
    #     dududu.recognize(draw) 
    #     cv2.imshow('Video', draw)
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break

    # video_capture.release()
    # cv2.destroyAllWindows()
    draw=cv2.imread("10.jpg")
    dududu.recognize(draw)
    cv2.imwrite("result.jpg",draw)
