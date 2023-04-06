# from time import sleep
import os.path

from tqdm import tqdm
a = 0
for i in tqdm(range(1000000)):
    pass

fold_name = './train_path/p_image/1.png'
name = os.path.basename(fold_name).split('.')[0]
print(name)

# 导入必要的库
import numpy as np
from sklearn.linear_model import LinearRegression

# 准备数据集
X = np.array([[1], [2], [3], [4], [5]])  # 自变量
y = np.array([2, 4, 5, 4, 5])  # 因变量

# 创建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X, y)

# 预测
X_test = np.array([[6], [7], [8]])  # 测试数据集
y_pred = lr.predict(X_test)  # 预测结果

# 输出结果
print('Coefficients:', lr.coef_)  # 输出模型的系数
print('Intercept:', lr.intercept_)  # 输出模型的截距
print('Predictions:', y_pred)  # 输出预测结果
