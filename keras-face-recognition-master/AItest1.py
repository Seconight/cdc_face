import grpc
import AIService_pb2 as pb2
import AIService_pb2_grpc as pb2_grpc
def run():
    # 连接 rpc 服务器测试文档
    channel = grpc.insecure_channel('localhost:50051')
    # 调用 rpc 服务
    stub = pb2_grpc.FaceStub(channel)
    response = stub.FaceDetect(pb2.DetectRequest(Id='1',shouldStudents='0121810880701,0121810880702,0121810880703,0121810880704,0121810880705,0121810880706,0121810880707,0121810880708,0121810880709,0121810880710,0121810880711,0121810880712,0121810880713,0121810880714,0121810880715,0121810880716,0121810880717,0121810880718,0121810880719,0121810880720,0121810880721,0121810880722,0121810880723,0121810880724,0121810880725,0121810880726,0121810880727,0121810880728,0121810880729,0121810880730,0121810880731,0121810880732,0121810880733,0121810880734,0121810880735,0121810880736,0121810880737,0121810880738,0121810880739,0121810880740,0121810880741,0121810880742,0121810880743,0121810880744,0121810880745,0121810880746,0121810880747,0121810880748,0121810880749,0121810880750,0121810880751,0121810880752,0121810880753,0121810880754,0121810880755,0121810880756,0121810880757,0121810880758,0121810880759,0121810880760,0121810880761,0121810880762,0121810880763,0121810880764,0121810880765,0121810880766,0121810880767,0121810880768,0121810880769,0121810880770,0121810880771,0121810880772,0121810880773,0121810880774,0121810880775,0121810880776,0121810880777,0121810880778,0121810880779,0121810880780,0121810880781,0121810880782,0121810880783,0121810880784,0121810880785,0121810880786,0121810880787,0121810880788,0121810880789,0121810880790,0121810880791,0121810880792,0121810880793,0121810880794,0121810880795,0121810880796,0121810880797,0121810880798,0121810880799,01218108807100,01218108807101,01218108807102'))
    print("client received: " + response.actualStudents)
    # stu='01218108807'
    # student=''
    # for i in range(100,102):
    #     if i < 9:
    #         student=stu+'0'+str(i+1)
    #     else:
    #         student=stu+str(i+1)
        
    #     response = stub.FaceRecognize(pb2.RecognizeRequest(studentId=student))
    #     print("client received: " + response.encodings)

if __name__ == '__main__':
    run()