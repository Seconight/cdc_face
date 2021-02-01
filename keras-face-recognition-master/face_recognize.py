import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1

class face_rec():
    def __init__(self):
<<<<<<< Updated upstream
        self.mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
        self.threshold = [0.5,0.8,0.9]  #门限
        self.known_face_encodings=[]    #编码后的人脸
        self.known_face_names=[]    #编码后的人脸的名字
        #载入facenet将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)
        # 从学生encoding文件中读取encoding信息
        f=open('./shouldStudents.txt','r')
        students=""
        s=f.readlines()
        #print(s)
        for each in s:
            students=students+each
        f.close()
        student=students.split(";")
        for each in student:
            studnetId=each[:13]
            studentEncodings=each[14:]
            face_encoding=studentEncodings.split(",")
            newencoding=[]
            for each in face_encoding:
                newencoding.append(float(each))
            # 存进已知列表中
            # print(newencoding)
            self.known_face_encodings.append(newencoding)
            self.known_face_names.append(studnetId)

=======
            self.known_face_encodings=[]    #编码后的人脸
            self.known_face_names=[]    #编码后的人脸的名字
            self.mtcnn_model = mtcnn()
            self.threshold = [0.5,0.8,0.9]
             #载入facenet将检测到的人脸转化为128维的向量
            self.facenet_model = InceptionResNetV1()
            model_path = './model_data/facenet_keras.h5'
            self.facenet_model.load_weights(model_path)
            # 从学生encoding文件中读取encoding信息
            f=open('./shouldStudents.txt','r')
            students=""
            s=f.readlines()
            #print(s)
            for each in s:
                students=students+each
            f.close()
            student=students.split(";")
            for each in student:
                studnetId=each[:13]
                studentEncodings=each[14:]
                face_encoding=studentEncodings.split(",")
                newencoding=[]
                for each in face_encoding:
                    newencoding.append(float(each))
                # 存进已知列表中
                self.known_face_encodings.append(newencoding)
                self.known_face_names.append(studnetId)
            print(self.known_face_names)
>>>>>>> Stashed changes

    def recognize(self,draw):
        #人脸识别
        #先定位，再进行数据库匹配
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

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
            matches = utils.compare_faces(self.known_face_encodings, face_encoding )
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                print(name)
                i=0
            face_names.append(name)
        actualStudent=""
        absebtStudent=""
        print(face_names)
        print(self.known_face_names)
        for name in self.known_face_names:
            if name in face_names:
                actualStudent=actualStudent+name+','
            else:
                absebtStudent=absebtStudent+name+','
<<<<<<< Updated upstream
=======
        # if(actualStudent.__len__!=0):
        #     actualStudent=actualStudent[0:actualStudent.__len__]
        # if(absebtStudent.__len__!=0):
        #     absebtStudent=absebtStudent[0:absebtStudent.__len__]
>>>>>>> Stashed changes
        if(len(actualStudent)!=0):
            actualStudent=actualStudent[0:len(actualStudent)-1]
        if(len(absebtStudent)!=0):
            absebtStudent=absebtStudent[0:len(absebtStudent)-1]
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
        f=open('./actualStudent.txt','w')
        f.write(actualStudent)
        f.close()
        f=open('./absentStudent.txt','w')
        f.write(absebtStudent)
        f.close()

        rectangles = rectangles[:,0:4]

        #画框
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 
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
    draw=cv2.imread("attendance.jpg")
    dududu.recognize(draw)
 
