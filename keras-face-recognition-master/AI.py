import socket
import sys
import cv2
import os
import threading
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
import time
from models import MobileFaceNet
import glob
import argparse
import tensorflow as tf
from utils.utils import detect_face, align_face
import piexif
def extract_oneface(self,image, marigin=16):
    # detecting faces
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    rectangles= self.mtcnn_model.detectFace(image, self.threshold)
    print(len(rectangles))
    if len(rectangles)!=1:
        return None
    for rectangle in rectangles:
        bounding_boxes = {
                'box': [int(rectangle[0]), int(rectangle[1]),
                        int(rectangle[2]-rectangle[0]), int(rectangle[3]-rectangle[1])],
                'confidence': rectangle[4],
                'keypoints': {
                        'left_eye': (int(rectangle[5]), int(rectangle[6])),
                        'right_eye': (int(rectangle[7]), int(rectangle[8])),
                        'nose': (int(rectangle[9]), int(rectangle[10])),
                        'mouth_left': (int(rectangle[11]), int(rectangle[12])),
                        'mouth_right': (int(rectangle[13]), int(rectangle[14])),
                }
        }

        bounding_box = bounding_boxes['box']
        keypoints = bounding_boxes['keypoints']

        # align face and extract it out
        align_image = align_face(image, keypoints)
        align_image = cv2.cvtColor(align_image ,cv2.COLOR_RGB2BGR)

        xmin = max(bounding_box[0] - marigin, 0)
        ymin = max(bounding_box[1] - marigin, 0)
        xmax = min(bounding_box[0] + bounding_box[2] + marigin, w)
        ymax = min(bounding_box[1] + bounding_box[3] + marigin, h)

        crop_image = align_image[ymin:ymax, xmin:xmax, :]
        # "just need only one face"
        return crop_image
#人脸识别对象类
class face_rec():
    def __init__(self):
        print('开始加载mtcnn权重。。。')
        self.mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
        print("mtcnn权重加载完毕！！！")
        self.threshold = [0.5,0.8,0.9]  #门限
        print('开始加载MobileFaceNet权重。。。')
        self.facenet_model = MobileFaceNet()
        print("MobileFaceNet权重加载完毕！！！")

    #学生编码
    def encoding(self, studentId):

        #返回示例：第一位代表成功或者出错（1/0），后面接的详细信息
        #比如返回1              代表成功
        #       01              代表'学生图片未找到或不存在'    
        #       02+图片路径     代表'图片检测到人脸数量不对'后面紧接的是图片路径

        
        # 打开对应学生文件夹，获取文件路径列表
        image_path = "./userFace/%s" %(studentId)
        image_list = glob.glob(image_path + "/*.jpg")
        if len(image_list)==0 :
            return '01'
        #初始化embedding
        embeddings = []
        flag=0
        faild_path=[]
        #编列文件列表进行人脸embedding
        for im_path in image_list:
            #调用piexif库的remove函数直接去除exif信息。
            piexif.remove(im_path)
            #读取图片文件
            image = cv2.imread(im_path)
            #检测人脸
            print(im_path+'detect')
            face = extract_oneface(self,image)
            
            if face is None:
                flag=1
                faild_path.append(im_path)
                continue
                
            #人脸embedding
            embeddings.append(self.facenet_model(face))

        #将人脸embedding保存到文件里
        embedding = np.concatenate(embeddings, 0).mean(0).flatten()
        np.save("./userFace/%s/%s" %(studentId, studentId), embedding)
        if flag==0:
            return '1'
        else:
            errInfo='02'
            for path in faild_path:
                errInfo=errInfo+path+'&'
            return errInfo[:-1]
    #学生识别
    def recognize(self,id,students):#students表示学生
        #返回示例：第一位代表成功或者出错（1/0），后面接的详细信息
        #比如返回1              代表成功
        #       01              代表'当前课程没有学生'    
        #       02+图片路径     代表'图片没找到或不存在'后面紧接的是图片路径
        #
        studentsList=[]
        if len(students)==0:
            return '01'
        else:
            studentsList=students.split(',')
        database_embeddings = {p:np.load("./userFace/%s/%s.npy" %(p, p)) for p in studentsList}
        face_names = ''
        for root, ds, fs in os.walk(".\\attendance\\"+id):#获得文件夹下所有文件
            for f in fs:
                #读取文件
                fullname = os.path.join(root, f)#文件全名
                #调用piexif库的remove函数直接去除exif信息。
                piexif.remove(fullname)
                print('多余信息已去除')
                print('start reading '+fullname+'...................')
                draw=cv2.imread(fullname)
                if draw is None:
                    return '02'+fullname  
                print('finish reading imag')
                org_image = draw.copy()
                #人脸识别
                #先定位，再进行数据库匹配
                height,width,_ = np.shape(draw)
                draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
                image_h, image_w, _ = draw.shape
                new_h, new_w = image_h, image_w
                print("开始检查")
                # 检测人脸
                start1 = time.time()
                rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
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
                indexList=[]
                nameList=[]
                distList=[]
                i=0
                marigin = 16

                for rectangle in rectangles:
                    bounding_boxes = {
                            'box': [int(rectangle[0]), int(rectangle[1]),
                                    int(rectangle[2]-rectangle[0]), int(rectangle[3]-rectangle[1])],
                            'confidence': rectangle[4],
                            'keypoints': {
                                    'left_eye': (int(rectangle[5]), int(rectangle[6])),
                                    'right_eye': (int(rectangle[7]), int(rectangle[8])),
                                    'nose': (int(rectangle[9]), int(rectangle[10])),
                                    'mouth_left': (int(rectangle[11]), int(rectangle[12])),
                                    'mouth_right': (int(rectangle[13]), int(rectangle[14])),
                            }
                    }

                    bounding_box = bounding_boxes['box']
                    keypoints = bounding_boxes['keypoints']

                    cv2.circle(org_image,(keypoints['left_eye']),   2, (255,0,0), 3)
                    cv2.circle(org_image,(keypoints['right_eye']),  2, (255,0,0), 3)
                    cv2.circle(org_image,(keypoints['nose']),       2, (255,0,0), 3)
                    cv2.circle(org_image,(keypoints['mouth_left']), 2, (255,0,0), 3)
                    cv2.circle(org_image,(keypoints['mouth_right']),2, (255,0,0), 3)
                    cv2.rectangle(org_image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                            (0,255,0), 2)
                    # align face and extract it out
                    align_image = utils.align_face(draw_rgb, keypoints)

                    xmin = max(bounding_box[0] - marigin, 0)
                    ymin = max(bounding_box[1] - marigin, 0)
                    xmax = min(bounding_box[0] + bounding_box[2] + marigin, new_w)
                    ymax = min(bounding_box[1] + bounding_box[3] + marigin, new_h)

                    crop_image = align_image[ymin:ymax, xmin:xmax, :]
                    if crop_image is not None:
                            t1 = time.time()
                            embedding = self.facenet_model(crop_image)
                            person,dist = utils.recognize_face(embedding,studentsList,database_embeddings)
                            if person!='Unknown':
                                face_names=face_names+person+','
                                indexList.append(i)
                                nameList.append(person)
                                distList.append(dist)
                            t2 = time.time()
                            print("recognize time: %.2fms" %((t2-t1)*1000))
                    i=i+1
                bestNameList=[]
                
                #找到识别出的重复人脸中距离最近的标出
                for k in range(len(nameList)):
                    #找出name相同的
                    # print('bestList:')
                    # print('len:'+str(len(bestNameList)))
                    # for i in range(len(bestNameList)):
                    #     print(bestNameList[i])
                    sameNameindex=[]
                    # print('当前人脸'+nameList[k])
                    if nameList[k] in bestNameList:
                        continue
                    else:
                        for n in range(k,len(nameList)):
                            if nameList[n]==nameList[k]:
                                sameNameindex.append(n)
                    # print('namelist:')
                    # for x in sameNameindex:
                    #     print(x)
                    min1=2.0
                    minIndex=0
                    for a in range(len(sameNameindex)):
                        if float(distList[sameNameindex[a]])<float(min1):
                            min1=distList[sameNameindex[a]]
                            minIndex=sameNameindex[a]
                    #画学号
                    # print(nameList[minIndex])
                    bestNameList.append(nameList[minIndex])
                    cv2.putText(org_image, nameList[minIndex], (int(rectangles[indexList[minIndex]][0]), int(rectangles[indexList[minIndex]][1])-2), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.75, (0, 255, 0), 2, lineType=cv2.LINE_AA)

                                                                                       
                cv2.imwrite(fullname, org_image)
                print(fullname+'write finish')
        if(len(face_names)!=0):
            return face_names[:-1]
        else:
            return ''
        

                #对检测到的人脸进行编码
        #         face_encodings = [] #人脸编码列表
        #         for rectangle in rectangles:
        #             #人脸五个关键点
        #             landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
        #             #截出人脸
        #             crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        #             crop_img = cv2.resize(crop_img,(160,160))
        #             #人脸矫正
        #             new_img,_ = utils.Alignment_1(crop_img,landmark)
        #             new_img = np.expand_dims(new_img,0)
        #             #计算128维特征向量并保存在列表中
        #             face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
        #             face_encodings.append(face_encoding)
        #         i=0

        #         face_names = []
        #         for face_encoding in face_encodings:
        #             # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
        #             matches = utils.compare_faces(known_face_encodings, face_encoding )
        #             name = "Unknown"
        #             # 找出距离最近的人脸
        #             face_distances = utils.face_distance(known_face_encodings, face_encoding)
        #             # 取出这个最近人脸的评分
        #             best_match_index = np.argmin(face_distances)
        #             if matches[best_match_index]:
        #                 name = known_face_names[best_match_index]
        #                 print(name)
        #                 i=0
        #             face_names.append(name)

        #         actualStudent=""

        #         print('当前检测学生:')
        #         print(face_names)
        #         for name in known_face_names:
        #             if name in face_names:
        #                 actualStudent=actualStudent+name+','
        #         if(len(actualStudent)!=0):
        #             actualStudent=actualStudent[0:len(actualStudent)-1]                   
        #         rectangles = rectangles[:,0:4]

        #         actualStu = actualStu + actualStudent + ","
        #         print('2:'+actualStudent)
        # if(len(actualStu)!=0):
        #     actualStu=actualStu[0:len(actualStu)-1] 
        # return actualStu
    