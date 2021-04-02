import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

# Parameter
time_step = 5
num_data_type = 3
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
num_input = num_one_joint_data*num_joint + time_step # joint data + ee_acc data
num_output = 2
threshold = 0.5

tf.compat.v1.disable_eager_execution()

# Model 1
tf.compat.v1.reset_default_graph()
sess1 = tf.compat.v1.Session()

new_saver1 = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
new_saver1.restore(sess1, 'model/model.ckpt')

graph1 = tf.compat.v1.get_default_graph()
x1 = graph1.get_tensor_by_name("input:0")
y1 = graph1.get_tensor_by_name("output:0")
drop_out_rate = graph1.get_tensor_by_name("drop_out_rate:0")
is_train1 = graph1.get_tensor_by_name("is_train:0")
hypothesis1 = graph1.get_tensor_by_name("ConcatenateNet/hypothesis:0")

file_name = '../data_tro/TestingDataCollision6.csv'
TestData = pd.read_csv(file_name).to_numpy().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
hypo1  =  sess1.run(hypothesis1, feed_dict={x1: X_Test, drop_out_rate: 0.0, is_train1:False})
t = np.arange(0,0.001*len(JTS),0.001)

collision_pre = 0
collision_cnt = 0
collision_time = 0
detection_time_NN = []
detection_time_JTS = []
detection_time_DoB = []
collision_status = False
NN_detection = False
JTS_detection = False
DoB_detection = False
collision_fail_cnt_NN = 0
collision_fail_cnt_JTS = 0
collision_fail_cnt_DoB = 0

continuous_filter_thres = 0
continuous_filter = 0
is_continuous_filter_on = False

for i in range(len(JTS)):
    if (Y_Test[i,0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    

    if (hypo1[i,0] > threshold and is_continuous_filter_on is False):
        is_continuous_filter_on = True
        continuous_filter += 1
    if (hypo1[i,0] > threshold and is_continuous_filter_on is True):
        continuous_filter += 1
    if (hypo1[i,0] < threshold):
        is_continuous_filter_on = False
        continuous_filter = 0
    if (collision_status == True and NN_detection == False):
        if(continuous_filter > continuous_filter_thres):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    # if (collision_status == True and NN_detection == False):
    #     if(hypo1[i,0] > threshold):
    #         NN_detection = True
    #         detection_time_NN.append(t[i] - collision_time)

    if (collision_status == True and JTS_detection == False):
        if(JTS[i] == 1):
            JTS_detection = True
            detection_time_JTS.append(t[i] - collision_time)
    
    if (collision_status == True and DoB_detection == False):
        if(DOB[i] == 1):
            DoB_detection = True
            detection_time_DoB.append(t[i] - collision_time)

    if (Y_Test[i,0] == 0 and collision_pre == 1):
        collision_status = False
        if(NN_detection == False):
            detection_time_NN.append(0.0)
            collision_fail_cnt_NN = collision_fail_cnt_NN+1
        if(JTS_detection == False):
            detection_time_JTS.append(0.0)
            collision_fail_cnt_JTS = collision_fail_cnt_JTS+1
        if(DoB_detection == False):
            detection_time_DoB.append(0.0)
            collision_fail_cnt_DoB = collision_fail_cnt_DoB+1
    collision_pre = Y_Test[i,0]
print('----------------------------------------')
print('Total collision: ', collision_cnt)
print('NN Failure: ', collision_fail_cnt_NN)
print('JTS Failure: ', collision_fail_cnt_JTS)
print('DOB Failure: ', collision_fail_cnt_DoB)
print('NN Detection Time: ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
print('JTS Detection Time: ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
print('DOB Detection Time: ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))

file_name = '../data_tro/TestingDataFree6.csv'
TestDataFree = pd.read_csv(file_name).to_numpy().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
JTSFree = TestDataFree[:,num_input]
DOBFree = TestDataFree[:,num_input+1]
hypofree1  =  sess1.run(hypothesis1, feed_dict={x1: X_TestFree, drop_out_rate: 0.0, is_train1:False})
t_free = np.arange(0,0.001*len(JTSFree),0.001)
NN_FP_time = []
NN_FP = 0
JTS_FP_time = []
JTS_FP = 0
DOB_FP_time = []
DOB_FP = 0
# for j in range(len(Y_TestFree)):
#     if (hypofree1[j,0] > threshold and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
#         NN_FP_time.append(t_free[j])
#         NN_FP = NN_FP + 1
#     if (JTSFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
#         JTS_FP_time.append(t_free[j])
#         JTS_FP = JTS_FP + 1
#     if (DOBFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
#         DOB_FP_time.append(t_free[j])
#         DOB_FP = DOB_FP + 1

continuous_filter = 0
is_continuous_filter_on = False
for j in range(len(Y_TestFree)):
    if (hypofree1[j,0] > threshold and is_continuous_filter_on is False):
        is_continuous_filter_on = True
        continuous_filter += 1
    if (hypofree1[j,0] > threshold and is_continuous_filter_on is True):
        continuous_filter += 1
    if (hypofree1[j,0] < threshold):
        is_continuous_filter_on = False
        continuous_filter = 0
    if (continuous_filter > continuous_filter_thres and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
        NN_FP_time.append(t_free[j])
        NN_FP = NN_FP + 1


    if (JTSFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
        JTS_FP_time.append(t_free[j])
        JTS_FP = JTS_FP + 1
    if (DOBFree[j] == 1 and np.equal(np.argmax(Y_TestFree[j,:]), 1)):
        DOB_FP_time.append(t_free[j])
        DOB_FP = DOB_FP + 1


print('----------------------------------------')
print("NN FP Time: ")
for k in range(NN_FP-1):
    del_time = abs(NN_FP_time[k+1]- NN_FP_time[k])
    if(del_time > 0.5):
        print(del_time)
print("JTS FP Time: ")
for k in range(JTS_FP-1):
    del_time = abs(JTS_FP_time[k+1]- JTS_FP_time[k])
    if(del_time > 0.5):
        print(del_time)
print("DOB FP Time: ")
for k in range(DOB_FP-1):
    del_time = abs(DOB_FP_time[k+1]- DOB_FP_time[k])
    if(del_time > 0.5):
        print(del_time)