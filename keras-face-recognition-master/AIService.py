from concurrent import futures
import time
import grpc
import os
import AIService_pb2 as pb2
import AIService_pb2_grpc as pb2_grpc
from AI import face_rec
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#创建人脸识别类对象
hhh = face_rec() 
# 实现 proto 文件中定义的 GreeterServicer
class Face(pb2_grpc.FaceServicer):
    # 实现 proto 文件中定义的 rpc 调用   
    def FaceDetect(self, request, context):
        return pb2.DetectReply(actualStudents =hhh.center_recognize(request.Id,request.shouldStudents))
    def FaceRecognize(self,request,context):
        return pb2.RecognizeReply(encodings=hhh.encoding(request.studentId))

def serve():
    # 启动 rpc 服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    pb2_grpc.add_FaceServicer_to_server(Face(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("server started!")
    try:
        while True:
            time.sleep(60*60*24) # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
