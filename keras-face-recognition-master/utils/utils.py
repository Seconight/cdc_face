import sys
from operator import itemgetter
import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
persons = os.listdir("userFace")

#计算原始输入图像每一次缩放的比例（构建图像金字塔）
def calculateScales(img):   #img为传入的图像
    copy_img = img.copy()   #复制图像
    pr_scale = 1.0  #初始缩放比例，若未缩放则为1
    h,w,_ = copy_img.shape  #获得图像的高和宽

    #开始进行缩放防止输入到Pnet的图像过大或者过小
    if min(w,h)>500:
        #当宽和高最小值大于100的时候将大小缩小到500上
        pr_scale = 500.0/min(h,w)   #缩小比例
        w = int(w*pr_scale) #缩小后的的宽
        h = int(h*pr_scale) #缩小后的高
    elif max(w,h)<500:
        #当宽和高最大值小于500的时候将大小放大到500上
        pr_scale = 500.0/max(h,w)   #放大比例
        w = int(w*pr_scale) #放大后的宽
        h = int(h*pr_scale) #放大后的高

    scales = [] #建立scales的空列表存放图像金字塔的缩放比例
    factor = 0.709  #缩放因子(固定值)
    factor_count = 0    #缩放次数
    minl = min(h,w) #高和宽中较小值
    while minl >= 12:   #一直使用缩放因子进行缩放直到当最小值小于12的时候退出循环
        #第一次缩放是在上面缩放到500，若原图为700*1000的，那么pr_scale=500/700
        #scales的第一个值也就是(500/700)*(0.709^0)=500/700
        #以后的每次缩放使用缩放因子进行缩放,那么缩放比例就是scales[i]=(500/700)*(0.709^i),i从0开始计数数值上等于factor_count
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor  #缩放后的最小值
        factor_count += 1   #缩放次数加一
    return scales   #返回缩放比例的列表


#对Pnet处理后的结果进行处理
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    #原始权重是caffe上训练的，yx轴是交换的，这里将y和x进行转换
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    stride = 0  #Pnet会对图像进行压缩，stride表示图像的压缩比例
    #stride略等于2
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1)
    (x,y) = np.where(cls_prob>=threshold)   #取出大于门限的网格点表示人脸存在可能性大的网格

    boundingbox = np.array([x,y]).T
    #找到对应原图的位置，将在缩放后的图像上检测到的网格映射到输入Pnet的图像上（通过stride），并进一步映射到原图上（通过scale）
    bb1 = np.fix((stride * (boundingbox) + 0 ) * scale) #左上角
    bb2 = np.fix((stride * (boundingbox) + 11) * scale) #右下角
    boundingbox = np.concatenate((bb1,bb2),axis = 1)
    
    #这一部分看不懂有知道的写一下
    #dx1，dx2为左上角网格偏移坐标；dx3，dx4为右下角网格偏移坐标
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T  #得到偏移量
    boundingbox = boundingbox + offset*12.0*scale   #变换到原图

    #进行堆叠将矩形变换成正方形方便Onet处理
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)    
    pick = []
    #对坐标进行限制不能超出图片范围
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)    #进行非极大值抑制剔除掉重合率较高的框


#将长方形调整为正方形
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]   #求出矩形宽
    h = rectangles[:,3] - rectangles[:,1]   #求出矩形高
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles


#非极大抑制剔除掉重合率较高的框
def NMS(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)    #转化成numpy数组
    #取出左上右下两个点坐标
    x1 = boxes[:,0] #取boxes每个元素的第一项
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4] #得分
    area = np.multiply(x2-x1+1, y2-y1+1)    #计算矩形区域面积
    I = np.array(s.argsort())   #对得分进行升序排序
    pick = []
    while len(I)>0:
        #得到相交区域
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1]表示I的倒数第一个元素的内容 拥有最高得分, I[0:-1]->othersI的第一个到倒数第二个
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        #计算相交区域的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h 
        #计算IoU：重叠面积/（面积1+面积2-重叠面积）
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        #保留得分最高的Box
        pick.append(I[-1])
        #保留IoU小于阈值的Box
        I = I[np.where(o<=threshold)[0]]
    #返回结果
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


#对Rnet处理后的结果进行处理
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    #筛选出人脸置信度大于门限的
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    #转化成numpy数组
    rectangles = np.array(rectangles)
    
    #原始人脸框位置
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    #这一步在干啥还不知道！！！
    sc  = np.array([prob[pick]]).T
    #对原始人脸框进行调整的参数
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    #原始人脸框宽和高
    w   = x2-x1
    h   = y2-y1
    #调整后的人脸框的位置
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    #进行堆叠并转换成正方形方便Onet处理
    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = rect2square(rectangles)

    #对坐标进行限制不能超出图片范围
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)    #非极大值抑制


#对Onet处理后的结果进行处理
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    #筛选出人脸置信度大于门限的框
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)   #转化成numpy数组
    #原始人脸框位置
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]

    sc  = np.array([prob[pick]]).T
    #对原始人脸框进行调整的参数
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    #原始人脸框宽和高
    w   = x2-x1
    h   = y2-y1
    
    #人脸的五个特征点位置
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T

    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T

    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T

    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T

    #调整后的人脸框的位置
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    #进行堆叠
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)

    #对坐标进行限制不能超出图片范围
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3)    #非极大值抑制


#人脸对齐
def Alignment_1(img,landmark):
    #计算眼睛连线相对于水平线的倾斜角
    if landmark.shape[0]==68:
        x = landmark[36,0] - landmark[45,0]
        y = landmark[36,1] - landmark[45,1]
    elif landmark.shape[0]==5:
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]
    #计算弧度值
    if x==0:
        angle = 0
    else: 
        angle = math.atan(y/x)*180/math.pi
    #求出旋转中心位置
    center = (img.shape[1]//2, img.shape[0]//2)
    #进行矫正
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0])) 
    #计算对齐过后五个特征点的位置
    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []    
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


#图片预处理高斯归一化
#归一化讲解见https://blog.csdn.net/program_developer/article/details/78637711
def pre_process(x):
    #这里在干什么我不知道！！！
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True) #求平均值
    std = np.std(x, axis=axis, keepdims=True)   #求标准差
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj    #归一化
    return y


#L2标准化
#公式见https://blog.csdn.net/ningyanggege/article/details/82840233
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output



#计算128特征值
def calc_128_vec(model,img):
    face_img = pre_process(img) #高斯归一化处理
    pre = model.predict(face_img)   #进行预测
    pre = l2_normalize(np.concatenate(pre)) #把预测的结果堆叠并进行L2标准化
    pre = np.reshape(pre,[128]) #将输入矩阵变为128行的矩阵
    return pre


#计算人脸距离
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1) #对矩阵的每一行求L2范数
                     #norm函数详解见https://blog.csdn.net/cjhxydream/article/details/108192497

#比较人脸
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance(known_face_encodings, face_encoding_to_check) 
    return list(dis <= tolerance)

#reshape
def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh))
    new_image = np.ones([size[1],size[0],3])*128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

#更改图片尺寸
def reshape_face(src_img):
    h,w,c=src_img.shape
    best_h=3000
    best_w=4000
    if(h>best_h or w>best_w):
        #调整宽和高至比例最接近4000*3200
        hb=best_h/h
        wb=best_w/w
        if(hb<=wb):
            img = cv2.resize(src_img, (int(w*hb), int(h*hb)))
            return img
        else:
            img=cv2.resize(src_img,(int(w*wb), int(h*wb)))
            return img

def recognize_face(q_emb,studentsList,database_embeddings, threshold=1.1):
    p_score = {}
    for person in database_embeddings:
        d_emb = database_embeddings[person]
        dist = caculateDist(d_emb, q_emb)
        if dist <= threshold:
            p_score[person] = dist
    if len(p_score) == 0:
        identity, dist = 'Unknown', 'None'
    else:
        p = sorted(p_score, key=lambda s: p_score[s])
        identity = p[0]  #学号
        dist = "%.4f" %p_score[identity]    #距离
    print("=> %s: %s" %(identity, dist), end="  ")
    return identity,dist

def align_face(image, keypoints, scale=1.0):
    eye_center = (
            (keypoints['left_eye'][0] + keypoints['right_eye'][0]) * 0.5,
            (keypoints['left_eye'][1] + keypoints['right_eye'][1]) * 0.5,
            )
    dx = keypoints['right_eye'][0] - keypoints['left_eye'][0]
    dy = keypoints['right_eye'][1] - keypoints['left_eye'][1]

    angle = cv2.fastAtan2(dy, dx)
    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
    rot_image = cv2.warpAffine(image, rot_matrix, dsize=(image.shape[1], image.shape[0]))
    return rot_image

def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    factor_count=0
    total_boxes=np.empty((0,9))
    points=np.empty(0)
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    m=12.0/minsize
    minl=minl*m
    # create scale pyramid
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data-127.5)*0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))
        out = pnet.predict_on_batch(img_y)
        out0 = np.transpose(out[0], (0,2,1,3))
        out1 = np.transpose(out[1], (0,2,1,3))

        boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox>0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick,:]
        regw = total_boxes[:,2]-total_boxes[:,0]
        regh = total_boxes[:,3]-total_boxes[:,1]
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,3,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = rnet.predict_on_batch(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]
        ipass = np.where(score>threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]
        if total_boxes.shape[0]>0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox>0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48,48,3,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = onet.predict_on_batch(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1,:]
        points = out1
        ipass = np.where(score>threshold[2])
        points = points[:,ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]

        w = total_boxes[:,2]-total_boxes[:,0]+1
        h = total_boxes[:,3]-total_boxes[:,1]+1
        points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
        points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
        if total_boxes.shape[0]>0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
            points = points[:,pick]

    return total_boxes, points

def bbreg(boundingbox,reg):
    """Calibrate bounding boxes"""
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox

def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride=2
    cellsize=12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])
    y, x = np.where(imap >= t)
    if y.shape[0]==1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y,x)]
    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
    return boundingbox, reg

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:,0].copy().astype(np.int32)
    y = total_boxes[:,1].copy().astype(np.int32)
    ex = total_boxes[:,2].copy().astype(np.int32)
    ey = total_boxes[:,3].copy().astype(np.int32)

    tmp = np.where(ex>w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w

    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data


def load_weights(model, weights_file):
    weights_dict = np.load(weights_file, encoding='latin1', allow_pickle=True).item()
    for layer_name in weights_dict.keys():
        layer = model.get_layer(layer_name)
        if "conv" in layer_name:
            layer.set_weights([weights_dict[layer_name]["weights"], weights_dict[layer_name]["biases"]])
        else:
            prelu_weight = weights_dict[layer_name]['alpha']
            try:
                layer.set_weights([prelu_weight])
            except:
                layer.set_weights([prelu_weight[np.newaxis, np.newaxis, :]])
    return True

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> '02h50m39s'
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)


def normalize(embedding):
    embedding = embedding / np.sqrt(np.sum(np.power(embedding, 2)) + 1e-9)
    return embedding

def caculateDist(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    dist = np.sum(np.square(diff),1)
    return dist

# def caculateScore(emb1, emb2):
    # diff = np.subtract(emb1, emb2)
    # dist = np.sqrt(np.power(diff, 2).sum())
    # return 1 - 0.5*dist

def caculateMetric(predict_issame, actual_issame):
    """
    shamlessly stolen code from https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L457
    """
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    prec = 0 if (tp+fp==0) else float(tp) / float(tp+fp)
    rec  = 0 if (tp+fn==0) else float(tp) / float(tp+fn)

    tpr  = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr  = 0 if (fp+tn==0) else float(fp) / float(fp+tn)

    acc  = float(tp+tn) / len(actual_issame)
    return prec, rec, tpr, fpr, acc


def evaluate(model, dataset, score_thresh, verbose=False):

    start_time = time.time()
    embeddings, gt_ids = [], []
    for _ in range(len(dataset)):

        images, labels = next(dataset)
        embeddings.append(model(images).numpy())
        gt_ids.append(labels.argmax(1))

    embeddings = np.concatenate(embeddings)
    gt_ids = np.concatenate(gt_ids)

    predict_issame, actual_issame = [], []
    for i in range(1, embeddings.shape[0]):
        embedding_1, embedding_2 = embeddings[i-1:i+1]
        score = caculateScore(embedding_1, embedding_2)

        if score >= score_thresh:
            predict_issame.append(True)
        else:
            predict_issame.append(False)

        gt_id1, gt_id2 = gt_ids[i-1:i+1]
        actual_issame.append(gt_id1 == gt_id2)

        if verbose:
            print("*----------------------------*")
            print(" ID1: %d, ID2: %d, Score: %.2f" %(gt_id1, gt_id2, score))

    prec, rec, tpr, fpr, acc = caculateMetric(predict_issame, actual_issame)
    print("\nElapsed time: %s | score_thres: %.2f | prec: %.4f | rec: %.4f | tpr: %.4f | fpr: %.4f | acc: %.4f\n"
        %(sec_to_hm(time.time()-start_time), score_thresh, prec, rec, tpr, fpr, acc))
    return prec, rec, tpr, fpr, acc
