echo 进行数据清洗
python wash.py
echo 进行训练数据生成
python txt_annotation.py
echo 开始训练...
python train.py
echo 生成个人特征向量
python AItest1.py