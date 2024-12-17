#T-ST-SSL：Hybrid Transformer and Spatial-Temporal Self-Supervised Learning for Long-term Traffic Prediction 
# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 
'''
在ST-SSL中添加了T-Transformer
'''

## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

## Model training and Evaluation

If the environment is ready, please run the following commands to train model on the specific dataset from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`.
```bash
>> cd ST-SSL
>> ./runme 0 PEMS04-6   # 0 gives the gpu id
```
##手动调试
'''
模型训练：configs文件夹下是PEMS04和PEMS08两个数据集下不同预测步长的参数设置。
	在main.py文件下选取数据集和参数（后缀为yaml文件），运行main.py文件，模型开始运行。

输入：交通速度信息 （B,T,N,C） B：batchsize T：历史时间步长 N：数据集路网节点个数 C：特征维度，此处为1，是交通速度
输出：预测交通速度 （B,T,N,C），其中T为1，是预测时间片上的预测速度。如30分钟处/45分钟处/60分钟处

代码的调整参数：在configs中，后缀名为yaml文件下是模型所有可设置的参数。

实验结果保存：参数自动保存在experimens文件下，并自动通过数据集名称-年月日-小时分钟秒来命名。
	      修改保存路径：在lib.utils中的代码中可以修改保存路径。
	      每个子文件夹下有三个子文件夹：best_model.pth：最优模型。 
	      run.log:运行日志，包括性能指标、资源利用率等
	      stats.pkl：包含用于分析和记录的统计信息的二进制文件。
'''

