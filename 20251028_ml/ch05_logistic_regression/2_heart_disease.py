import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

#1.加载数据集
dataset=pd.read_csv('../data/heart_disease.csv')
# print(dataset.head())
dataset.dropna(inplace=True)

#2.划分数据集
X=dataset.drop('是否患有心脏病',axis=1)
y=dataset['是否患有心脏病']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#3.特征工程
numerical_features=["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
categorical_features=["胸痛类型","静息心电图结果","峰值ST段的斜率","地中海贫血"]
binary_features=["性别","空腹血糖","运动性心绞痛"]

#4.创建一个转换器
column_transformer = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(),categorical_features),
        ("bin","passthrough",binary_features)
    ]
)

#5.特征转换
X_train=column_transformer.fit_transform(X_train)
X_test=column_transformer.transform(X_test)

#6.模型定义和训练
model=LogisticRegression()
model.fit(X_train,y_train)

#7.计算得分，评估模型
print(model.score(X_train,y_train))
