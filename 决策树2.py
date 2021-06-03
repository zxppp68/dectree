# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:48:17 2021

@author: 10359
"""

#import graphviz
import pydotplus
from sklearn import datasets # 导入方法类
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier #此时是分类树
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from IPython.display import Image, display

#加载数据，得到特征数据与分类数据
iris = datasets.load_iris()
iris_feature = iris.data 
iris_target = iris.target 


#划分训练集与测试集
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, 
                                                            iris_target, test_size=0.33, random_state=42)

target_train

#训练模型
dt_model = DecisionTreeClassifier(criterion='entropy',max_depth=5,splitter='best')
# 选择熵做损失函数，采用“最佳”分裂策略，控制最大深度为5防止过拟合
dt_model.fit(feature_train,target_train) 
# 使用训练集训练模型
predict_results = dt_model.predict(feature_test) 
# 使用模型对测试集进行预测

print('predict_results:', predict_results)
print('target_test:', target_test)


print('精确度=',accuracy_score(target_test,predict_results))

#可视化
#out_file=None直接把数据赋给image，不产生中间文件.dot
#filled=Ture添加颜色，rounded增加边框圆角
image = export_graphviz(dt_model, out_file=None,feature_names=iris.feature_names,
                       class_names=iris.target_names,filled=True,node_ids=True,rounded=True)
#graphviz.Source(image)
graph = pydotplus.graph_from_dot_data(image)
display(Image(graph.create_png()))