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
                #img = cv2.imread('./userFace/'+studentId+'/'+studentId+'_'+str(i)+'.jpg')    #读取对应的图像
                img = cv2.imread(fullname)
                # print(facePath)
                # img = cv2.imread(facePath)    #读取对应的图像
                #cv2.imshow(img)
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
                # 设置输出路径
                # encodingPath = './userFace/'+studentId+'/encoding'+studentId+'_'+str(i)+'.txt'
                # f=open(encodingPath,'w')
                # f.write(enc2str)
                # f.close()
                result = result + enc2str + ';'
        
        file = open('./encoding/resultE'+studentId,'a+')
        file.write(result)
        file.close()    
        

    #学生识别
    def recognize(self, id, students):#students表示学生
        
        #预加载文件
        # 从学生encoding文件中读取encoding信息
        # print(id)
        # f=open('./shouldStudents'+id+'.txt','r')
        # students=""
        # s=f.readlines()
        # print(s)
        # for each in s:
        #     students=students+each
        # f.close()
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
            # print(newencoding)
            known_face_encodings.append(newencoding)
            known_face_names.append(studnetId)
        print("学生加载")
        #result
        actualStu = ""
        start = time.time()
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
                # absebtStudent=""
                # acFile=open('./actualStudent.txt','a+')
                # acNames=acFile.readline()
                # acNames = acNames.split(',')
                # abFile=open('./absentStudent.txt','a+')
                # abNames = abFile.readline()
                # abNames = abNames.split(',')

                print('当前检测学生:')
                print(face_names)
                for name in known_face_names:
                    if name in face_names:
                        actualStudent=actualStudent+name+','
                
                # if(actualStudent.__len__!=0):
                #     actualStudent=actualStudent[0:actualStudent.__len__]
                # if(absebtStudent.__len__!=0):
                #     absebtStudent=absebtStudent[0:absebtStudent.__len__]
                if(len(actualStudent)!=0):
                    actualStudent=actualStudent[0:len(actualStudent)-1]                   
                rectangles = rectangles[:,0:4]

                #画框
                # for (left, top, right, bottom), name in zip(rectangles, face_names):
                #     cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 

                actualStu = actualStu + actualStudent + ","
                print('2:'+actualStudent)
        
        print("所有学生：")
        print(actualStu)
        if(len(actualStu)!=0):
            actualStu=actualStu[0:len(actualStu)-1] 
        print("正在将结果写入文件...")
        acFile = open('./recognize/resultR'+id,'a+')
        acFile.write(actualStu)
        acFile.close()
        print('文件输出完毕')
      



class taskThread (threading.Thread):
    def __init__(self, targetObj, id, studentId, students, function):    #理论上参数冗余
        print("init")
        threading.Thread.__init__(self)
        self.targetObj = targetObj  #表示face_rec对象
        self.students = students    #表示应到学生
        self.id = id                #任务ID
        self.function = function    #表示要做的功能
        self.studentId = studentId  #学生ID
        
    def run(self):  
        if self.function == '1' :
            self.targetObj.recognize(self.id, self.students)
            # print(type(self.result))
        elif self.function == '2':
            self.targetObj.encoding(self.studentId)
        else:
            print('错误！')

def threadListenAndRun():
    #建立socket对象
    serversocket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM) 
    #port
    port = 12345
    #创建服务器
    serversocket.bind(("localhost",port))
    serversocket.listen()
    print("准备就绪")
    while True:
        #accept
        clientsocket,address = serversocket.accept()
        print('干活了！干活了！')
        #接收多个参数
        ##表示当前id(毫秒级时间戳13位)
        currentId = clientsocket.recv(13)

        ##表示当前所选功能
        func = clientsocket.recv(1)

        ##表示学生ID
        stuId = clientsocket.recv(13)

        ##表示应到学生
        shouldS = clientsocket.recv(65535)

        #断开连接
        clientsocket.close()
        print('******************************************************************************************')
        print(str(currentId,'utf-8'))
        print(str(func,'utf-8'))
        print(str(stuId,'utf-8'))
        print(str(shouldS,'utf-8'))
        

        currentTask = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))
        # cu2 = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))
        currentTask.start()

def acceptClient(serverSocket):
    while True:
        clientSocket,address = serverSocket.accept()
        thread = threading.Thread(target=handleMessage,args=(clientSocket,))
        thread.start()

def handleMessage(clientSocket):
    print('干活了！干活了！')
    #接收多个参数
    ##表示当前id(毫秒级时间戳13位)
    currentId = clientSocket.recv(13)

    ##表示当前所选功能
    func = clientSocket.recv(1)

    ##表示学生ID
    stuId = clientSocket.recv(13)

    ##表示应到学生
    shouldS = clientSocket.recv(65535)

    #断开连接
    clientSocket.close()
    print('******************************************************************************************')
    print(str(currentId,'utf-8'))
    print(str(func,'utf-8'))
    print(str(stuId,'utf-8'))
    print(str(shouldS,'utf-8'))
        

    currentTask = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))
    # cu2 = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))
    currentTask.setDaemon(True)
    currentTask.start()

#####################################################################################
#PROGRAM BEGIN!
#####################################################################################

#设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#创建人脸识别类对象
dududu = face_rec() 

#**************************************************************************************
#建立socket对象
serversocket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM) 
#port
port = 12345
#创建服务器
serversocket.bind(("localhost",port))
serversocket.listen(100)
print("准备就绪")
#启动监听线程
start = threading.Thread(target=acceptClient,args=(serversocket,))
start.start()
# while True:
#     #accept
    
#     clientsocket,address = serversocket.accept()
#     print('干活了！干活了！')
#     #接收多个参数
#     ##表示当前id(毫秒级时间戳13位)
#     currentId = clientsocket.recv(13)

#     ##表示当前所选功能
#     func = clientsocket.recv(1)

#     ##表示学生ID
#     stuId = clientsocket.recv(13)

#     ##表示应到学生
#     shouldS = clientsocket.recv(65535)

#     #断开连接
#     clientsocket.close()
#     print('******************************************************************************************')
#     print(str(currentId,'utf-8'))
#     print(str(func,'utf-8'))
#     print(str(stuId,'utf-8'))
#     print(str(shouldS,'utf-8'))
    

#     currentTask = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))
#     # cu2 = taskThread(dududu, str(currentId,'utf-8'), str(stuId,'utf-8'), str(shouldS,'utf-8'),str(func,'utf-8'))

#     currentTask.start()
    

    