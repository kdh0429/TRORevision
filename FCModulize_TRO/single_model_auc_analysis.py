import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Parameter
time_step = 5
num_joint_data_type = 2
num_one_joint_data = time_step * num_joint_data_type
num_joint = 6
use_ee_acc_data = False
if use_ee_acc_data is False :
    num_input = num_one_joint_data*num_joint # joint data
else:
    num_input = num_one_joint_data*num_joint + time_step # joint data + ee_acc data
num_output = 2

tf.compat.v1.disable_eager_execution()

# Model 1
tf.compat.v1.reset_default_graph()
sess1 = tf.compat.v1.Session()

new_saver1 = tf.compat.v1.train.import_meta_graph('model/num_input_60_1617825852.ckpt.meta')
new_saver1.restore(sess1, 'model/num_input_60_1617825852.ckpt')

graph1 = tf.compat.v1.get_default_graph()
x1 = graph1.get_tensor_by_name("input:0")
y1 = graph1.get_tensor_by_name("output:0")
drop_out_rate = graph1.get_tensor_by_name("drop_out_rate:0")
is_train1 = graph1.get_tensor_by_name("is_train:0")
hypothesis1 = graph1.get_tensor_by_name("ConcatenateNet/hypothesis:0")

file_name = '../data_tro/TestingDataCollision1.parquet'
TestData = pd.read_parquet(file_name).to_numpy().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
hypo1  =  sess1.run(hypothesis1, feed_dict={x1: X_Test, drop_out_rate: 0.0, is_train1:False})

file_name = '../data_tro/TestingDataFree1.parquet'
TestDataFree = pd.read_parquet(file_name).to_numpy().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
hypofree1  =  sess1.run(hypothesis1, feed_dict={x1: X_TestFree, drop_out_rate: 0.0, is_train1:False})

Y_Test_All = np.concatenate((Y_Test, Y_TestFree), axis=0)
hypo_All = np.concatenate((hypo1, hypofree1), axis=0)

fpr, tpr, thresholds = metrics.roc_curve(Y_Test_All[:,0], hypo_All[:,0])
roc_auc = metrics.auc(fpr, tpr)
print("AUC: ",roc_auc)

display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()  
plt.show()      