import socket
import sys
import cv2
import os
import threading
import numpy as np
from net.mtcnn import mtcnn
from numpy.core.numeric import full
import utils.utils as utils
from net.inception import InceptionResNetV1

#人脸识别对象类
class face_rec():
    def __init__(self):
        self.mtcnn_model = mtcnn()  #创建mtcnn对象检测图片中的人脸
        self.threshold = [0.5,0.8,0.9]  #门限
        self.known_face_encodings=[]    #编码后的人脸
        self.known_face_names=[]    #编码后的人脸的名字
        #载入facenet将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)
        print("权重加载完毕")

    #获取人脸编码，返回encoding（后面有；）
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
                # 设置输出路径
                # encodingPath = './userFace/'+studentId+'/encoding'+studentId+'_'+str(i)+'.txt'
                # f=open(encodingPath,'w')
                # f.write(enc2str)
                # f.close()
                result = result + enc2str + ';'
        return result    



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
        print("学生加载")
        #result
        actualStu = ""

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
                rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
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
                # absebtStudent=""
                # acFile=open('./actualStudent.txt','a+')
                # acNames=acFile.readline()
                # acNames = acNames.split(',')
                # abFile=open('./absentStudent.txt','a+')
                # abNames = abFile.readline()
                # abNames = abNames.split(',')

                print(face_names)
                print(self.known_face_names)
                for name in self.known_face_names:
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
                
        if(len(actualStu)!=0):
            actualStu=actualStu[0:len(actualStu)-1]   
        return actualStu   

#设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

lala = face_rec()

# print(lala.encoding('0121810880277'))

print(lala.recognize(str(123),'0121810880204:0.07449723035097122,-0.0635077953338623,-0.08589383214712143,-0.013460466638207436,0.08448546379804611,0.04000774025917053,-0.047316018491983414,0.0022769817151129246,0.09547420591115952,0.012387043796479702,-0.11413606256246567,-0.033573973923921585,-0.09254490584135056,-0.15614156424999237,0.06847328692674637,0.09345153719186783,0.010843208059668541,0.010891927406191826,0.08093001693487167,-0.07824783772230148,-0.003526562824845314,-0.050174105912446976,-0.18515807390213013,-0.04572148248553276,-0.09439681470394135,0.0008491854532621801,0.05675202235579491,0.09711708873510361,-0.09600599855184555,0.06229166314005852,0.11840707808732986,0.10048507153987885,0.031233923509716988,-0.11930666118860245,-0.15503476560115814,-0.06822095066308975,-0.09573360532522202,-0.11166056990623474,0.06804901361465454,0.004575646482408047,-0.0764872208237648,-0.07850352674722672,-0.004103292711079121,-0.17855076491832733,0.016681812703609467,0.03896217793226242,-0.004194659646600485,0.08975771069526672,-0.07929855585098267,0.04648986831307411,-0.05455863103270531,-0.030710648745298386,0.0014487684238702059,0.18434830009937286,0.1389344334602356,-0.12622465193271637,-0.0677935928106308,-0.08330050855875015,0.08793055266141891,-0.026701902970671654,-0.13530300557613373,-0.09983081370592117,0.12000302225351334,0.1695191115140915,0.15086019039154053,-5.5660493671894073e-05,0.05754080042243004,-0.012601355090737343,-0.09165633469820023,-0.02829243242740631,-0.03002784587442875,-0.14145544171333313,0.08758557587862015,0.06911355257034302,-0.035742245614528656,0.15380412340164185,-0.08660664409399033,0.11580658704042435,0.04436510428786278,0.025475487112998962,-0.0949515774846077,-0.04114442691206932,-0.09148521721363068,-0.06363323330879211,0.10518871992826462,-0.2051757574081421,-0.12361117452383041,-0.058402322232723236,-0.07803061604499817,0.0890074297785759,0.15742766857147217,-0.009120302274823189,-0.07042606920003891,-0.05944380909204483,-0.12200270593166351,-0.018696840852499008,-0.19731009006500244,0.1268489509820938,-0.10828985273838043,-0.05385657772421837,-0.026966676115989685,-0.0036824403796344995,0.09004276245832443,-0.08519122749567032,-0.055640943348407745,0.023164764046669006,-0.18567146360874176,0.11144320666790009,-0.05077343434095383,0.024480754509568214,-0.038098499178886414,0.02863006666302681,-0.11898742616176605,-0.04856664314866066,0.043281715363264084,0.020992398262023926,-0.024090280756354332,-0.0244552344083786,-0.04029158875346184,0.05647753179073334,0.02737964130938053,-0.05988147482275963,0.07328982651233673,0.10984479635953903,0.0756780207157135,-0.1556977778673172,0.026927724480628967,-0.0263320654630661;0121810880207:0.0038461280055344105,-0.09137902408838272,-0.0604860894382,0.006215541623532772,-0.07307124137878418,-0.041468266397714615,-0.004412480164319277,-0.08760541677474976,-0.06359124928712845,-0.15574757754802704,-0.01884612813591957,0.018391340970993042,0.05407952517271042,-0.18021471798419952,0.11573641747236252,0.02993333712220192,0.023297686129808426,0.0036844895221292973,-0.0199102945625782,0.03261945769190788,-0.15616099536418915,0.01392586249858141,-0.11183992773294449,-0.049209363758563995,-0.04330701008439064,0.05824527144432068,0.011219031177461147,0.1261225789785385,-0.10031211376190186,-0.049790047109127045,0.10533689707517624,0.0547742061316967,-0.009803262539207935,-0.044511765241622925,0.0035714437253773212,0.021796762943267822,-0.06600166857242584,-0.06739397346973419,0.10977689921855927,0.06558770686388016,-0.09108429402112961,0.06284981966018677,0.04794022813439369,-0.12952882051467896,0.06225834786891937,-0.02852063998579979,0.014335913583636284,0.15058691799640656,-0.05739200487732887,0.0737784132361412,0.05619661882519722,0.02609652653336525,0.026637759059667587,0.0772685706615448,0.09059394896030426,0.05601828917860985,-0.10108039528131485,0.08956275135278702,0.014308604411780834,-0.030596276745200157,0.04399479553103447,-0.10137303173542023,-0.05011875554919243,0.23667384684085846,0.1922336220741272,0.0358012355864048,0.03585987910628319,0.031110292300581932,-0.09422128647565842,0.0028774114325642586,0.021996958181262016,-0.1298804134130478,0.05659550800919533,-0.05341499671339989,0.03216740861535072,0.06312631815671921,-0.10057538002729416,0.10371057689189911,-0.10507404059171677,-0.021494407206773758,-0.049355797469615936,0.026667406782507896,-0.09075070172548294,0.12752363085746765,0.11444651335477829,-0.25122109055519104,0.01932728849351406,-0.1199486255645752,-0.23450936377048492,-0.11397822946310043,0.07348941266536713,-0.10495571792125702,-0.08281420916318893,-0.12983591854572296,-0.07182206213474274,0.018261384218931198,-0.10812326520681381,0.04754147678613663,-0.04246808588504791,-0.04515421763062477,-0.0264921672642231,0.05298726260662079,0.058533620089292526,0.01791396178305149,-0.13430382311344147,0.049912407994270325,-0.03813978657126427,-0.002543714363127947,-0.009387641213834286,-0.1504587084054947,-0.021231619641184807,0.0845225378870964,0.05277116224169731,-0.03477681428194046,0.08335155248641968,0.15872254967689514,-0.02326297201216221,0.08716213703155518,-0.054736968129873276,0.11150901764631271,0.09443094581365585,-0.12722435593605042,0.21611548960208893,0.016253340989351273,0.06963738054037094,-0.19544164836406708,-0.02073698304593563,0.05084143951535225'))