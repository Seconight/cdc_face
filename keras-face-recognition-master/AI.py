import socket
import sys
import cv2
import os
import threading
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
import time
from net.inception import InceptionResNetV1
from fresh import centerDetect
#人脸识别对象类
class face_rec():
    def __init__(self):
        self.mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
        self.threshold = [0.5,0.8,0.9]  #门限
        
        #载入facenet将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)
        print("权重加载完毕")

    #学生编码
    def encoding(self, studentId):
        
        result = ""
        for root, ds, fs in os.walk(".\\userface\\"+studentId):#获得文件夹下所有文件
            for f in fs:
                # faceImage
                fullname = os.path.join(root, f)#文件全名
                print(fullname)
                img = cv2.imread(fullname)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)  # 利用facenet_model检测人脸
                start = time.time()
                # 转化成正方形
                rectangles = utils.rect2square(np.array(rectangles))
                # facenet要传入一个160x160的图片
                rectangle = rectangles[0]
                # 人脸的五个关键点
                landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
                # 截下人脸图
                crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = cv2.resize(crop_img,(160,160))
                # 进行人脸对齐
                new_img,_ = utils.Alignment_1(crop_img,landmark)
                # 增加维度
                new_img = np.expand_dims(new_img,0)
                # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
                face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
                enc2str = ",".join(str(li) for li in face_encoding.tolist())
                end = time.time()
                print("facenet用时" + str(start - end))
                result = result + enc2str + ';'
        
        return result   
    #学生识别
    def recognize(self,id,students):#students表示学生
        known_face_encodings=[]    #编码后的人脸
        known_face_names=[]    #编码后的人脸的名字
        student=students.split(";")

        for each in student:
            studnetId=each[:13]
            studentEncodings=each[14:]
            face_encoding=studentEncodings.split(",")
            newencoding=[]
            for each in face_encoding:
                newencoding.append(float(each))
            # 存进已知列表中
            known_face_encodings.append(newencoding)
            known_face_names.append(studnetId)
        print("学生加载")
        actualStu = ""
        print("id is "+id)
        for root, ds, fs in os.walk(".\\attendance\\"+id):#获得文件夹下所有文件
            for f in fs:
                #读取文件
                fullname = os.path.join(root, f)#文件全名
                print(fullname)
                draw=cv2.imread(fullname)
                #人脸识别
                #先定位，再进行数据库匹配
                height,width,_ = np.shape(draw)
                draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

                print("开始检查")
                # 检测人脸
                start1 = time.time()
                rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
                print("人间检测输出为")
                print(type(rectangles))
                print(rectangles)
                end1 = time.time()
                print("mtcnn time: "+str(end1-start1))
                print("检查完毕")
                if len(rectangles)==0:
                    continue
                
                
                # 转化成正方形并同时限制不能超出图像范围
                rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
                rectangles[:,0] = np.clip(rectangles[:,0],0,width)
                rectangles[:,1] = np.clip(rectangles[:,1],0,height)
                rectangles[:,2] = np.clip(rectangles[:,2],0,width)
                rectangles[:,3] = np.clip(rectangles[:,3],0,height)
                
                #对检测到的人脸进行编码
                face_encodings = [] #人脸编码列表
                for rectangle in rectangles:
                    #人脸五个关键点
                    landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
                    #截出人脸
                    crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                    crop_img = cv2.resize(crop_img,(160,160))
                    #人脸矫正
                    new_img,_ = utils.Alignment_1(crop_img,landmark)
                    new_img = np.expand_dims(new_img,0)
                    #计算128维特征向量并保存在列表中
                    face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
                    face_encodings.append(face_encoding)
                i=0

                face_names = []
                for face_encoding in face_encodings:
                    # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
                    matches = utils.compare_faces(known_face_encodings, face_encoding )
                    name = "Unknown"
                    # 找出距离最近的人脸
                    face_distances = utils.face_distance(known_face_encodings, face_encoding)
                    # 取出这个最近人脸的评分
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        print(name)
                        i=0
                    face_names.append(name)

                actualStudent=""

                print('当前检测学生:')
                print(face_names)
                for name in known_face_names:
                    if name in face_names:
                        actualStudent=actualStudent+name+','
                if(len(actualStudent)!=0):
                    actualStudent=actualStudent[0:len(actualStudent)-1]                   
                rectangles = rectangles[:,0:4]

                actualStu = actualStu + actualStudent + ","
                print('2:'+actualStudent)
        if(len(actualStu)!=0):
            actualStu=actualStu[0:len(actualStu)-1] 
        return actualStu
    def center_encoding(self, studentId):

        result = ""
        for root, ds, fs in os.walk(".\\userface\\"+studentId):#获得文件夹下所有文件
            for f in fs:
                # faceImage
                fullname = os.path.join(root, f)#文件全名
                print(fullname)
                img = cv2.imread(fullname)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)  # 利用facenet_model检测人脸
                
                #替换成centerface
                #rectangles = centerDetect(img)
                #false:centerface不能检测大人脸
                #md，直接给我跑蓝屏了

                start = time.time()
                # 转化成正方形
                rectangles = utils.rect2square(np.array(rectangles))
                # facenet要传入一个160x160的图片
                rectangle = rectangles[0]
                # 人脸的五个关键点
                landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
                # 截下人脸图
                crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = cv2.resize(crop_img,(160,160))
                # 进行人脸对齐
                new_img,_ = utils.Alignment_1(crop_img,landmark)
                # 增加维度
                new_img = np.expand_dims(new_img,0)
                # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
                face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
                enc2str = ",".join(str(li) for li in face_encoding.tolist())
                end = time.time()
                print("facenet用时" + str(start - end))
                result = result + enc2str + ';'
        
        return result   

    def center_recognize(self,id,students):
        known_face_encodings=[]    #编码后的人脸
        known_face_names=[]    #编码后的人脸的名字
        student=students.split(";")

        for each in student:
            studnetId=each[:13]
            studentEncodings=each[14:]
            face_encoding=studentEncodings.split(",")
            newencoding=[]
            for each in face_encoding:
                newencoding.append(float(each))
            # 存进已知列表中
            known_face_encodings.append(newencoding)
            known_face_names.append(studnetId)
        print("学生加载")
        actualStu = ""
        print("id is "+id)
        for root, ds, fs in os.walk(".\\attendance\\"+id):#获得文件夹下所有文件
            for f in fs:
                #读取文件
                fullname = os.path.join(root, f)#文件全名
                print(fullname)
                draw=cv2.imread(fullname)
                draw = utils.letterbox_image(draw, [3200, 3200])
                #draw = utils.reshape_face(draw)
                #人脸识别
                #先定位，再进行数据库匹配
                height,width,_ = np.shape(draw)
                draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

                print("开始检查")
                # 检测人脸
                start1 = time.time()
                #rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
                rectangles = centerDetect(draw);
                print("人间检测输出为")
                #print(type(rectangles))
                print(rectangles)
                end1 = time.time()
                print("center time: "+str(end1-start1))
                print("检查完毕")
                if len(rectangles)==0:
                    continue
                
                
                # 转化成正方形并同时限制不能超出图像范围
                rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
                rectangles[:,0] = np.clip(rectangles[:,0],0,width)
                rectangles[:,1] = np.clip(rectangles[:,1],0,height)
                rectangles[:,2] = np.clip(rectangles[:,2],0,width)
                rectangles[:,3] = np.clip(rectangles[:,3],0,height)
                
                #对检测到的人脸进行编码
                face_encodings = [] #人脸编码列表
                for rectangle in rectangles:
                    #人脸五个关键点
                    landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
                    #截出人脸
                    crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                    crop_img = cv2.resize(crop_img,(160,160))
                    #人脸矫正
                    new_img,_ = utils.Alignment_1(crop_img,landmark)
                    new_img = np.expand_dims(new_img,0)
                    #计算128维特征向量并保存在列表中
                    face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
                    face_encodings.append(face_encoding)
                i=0

                face_names = []
                print('*************has known*****************')
                print(known_face_encodings)
                for face_encoding in face_encodings:
                    # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
                    matches = utils.compare_faces(known_face_encodings, face_encoding )
                    print('**************face_encoding*************')
                    print(face_encoding)
                    name = "Unknown"
                    # 找出距离最近的人脸
                    face_distances = utils.face_distance(known_face_encodings, face_encoding)
                    print('face_distances')
                    print(face_distances)
                    # 取出这个最近人脸的评分
                    best_match_index = np.argmin(face_distances)
                    print('best index')
                    print(best_match_index)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        print(name)
                        i=0
                    face_names.append(name)

                actualStudent=""

                print('当前检测学生:')
                print(face_names)
                for name in known_face_names:
                    if name in face_names:
                        actualStudent=actualStudent+name+','
                if(len(actualStudent)!=0):
                    actualStudent=actualStudent[0:len(actualStudent)-1]                   
                rectangles = rectangles[:,0:4]

                actualStu = actualStu + actualStudent + ","
                print('2:'+actualStudent)
        if(len(actualStu)!=0):
            actualStu=actualStu[0:len(actualStu)-1] 
        return actualStu