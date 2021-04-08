import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import wandb
import os
import time
import pandas as pd
import argparse
import math
 
start_time = time.time()

# Parameters


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--use_wandb', type=str2bool, default=True)
parser.add_argument('--use_gpu', type=str2bool, default=False)
parser.add_argument('--use_ee_acc_data', type=str2bool, default=True)
parser.add_argument('--learning_rate', type=float, default=2e-6)
parser.add_argument('--training_epoch', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--drop_out', type=float, default=1.0)
parser.add_argument('--regularization_factor', type=float, default=0e-7)
parser.add_argument('--hidden_neuron', type=int, default=16)
parser.add_argument('--cross_entropy_weight', type=float, default=0.01)
parser.add_argument('--input_type', type=str, default="input5")
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--num_data_type', type=int, default=5)
parser.add_argument('--collision_ratio', type=float, default=0.37)

args = parser.parse_args()
# Init wandb
wandb_use = args.use_wandb  
if wandb_use == True:
    wandb.init(project="Dusan_2nd_Project", name="finetuning_test_with_robot3free_robot12col_"+args.input_type+"_epoch_"+str(args.training_epoch)+"_ver"+str(args.version), tensorboard=False)

# Number of Input/Output Data
time_step = 5
num_data_type = args.num_data_type
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
if args.use_ee_acc_data is False :
    num_input = num_one_joint_data*num_joint # joint data
    num_concatenate_node = 1*num_joint
else:
    num_input = num_one_joint_data*num_joint + 1* time_step # joint data + delta ee_acc data + delta current
    num_concatenate_node = 1*num_joint + 1
num_output = 2

# Hyper parameter Setting
learning_rate = args.learning_rate
training_epochs = args.training_epoch
batch_size = args.batch_size
drop_out = args.drop_out
regul_factor = args.regularization_factor
hidden_neurons = args.hidden_neuron
cross_entropy_weight = args.cross_entropy_weight
test_data_hz = 1000


# Tensorflow Setting
if args.use_gpu is False :
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16, log_device_placement=False)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = False
    sess = tf.Session(config=tf_config)
else :
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess = tf.Session()

sess.run(tf.global_variables_initializer())

new_saver = tf.train.import_meta_graph('model/model_test_robot12_input5_v5.ckpt.meta')
new_saver.restore(sess, 'model/model_test_robot12_input5_v5.ckpt')
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("m1/input:0")
Y = graph.get_tensor_by_name("m1/output:0")
keep_prob = graph.get_tensor_by_name("m1/keep_prob:0")
is_train = graph.get_tensor_by_name("m1/is_train:0")
logits = graph.get_tensor_by_name("m1/ConcatenateNet/logits:0")
hypothesis = graph.get_tensor_by_name("m1/ConcatenateNet/hypothesis:0")
optimizer = graph.get_operation_by_name("m1/Adam")
learning_rate_tensor = graph.get_tensor_by_name("m1/Adam/learning_rate:0")

# Log Configuration
if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out = drop_out
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.time_step = time_step

# Parse .tfrecord Data
def parse_proto(example_proto):
  features = {
    'X': tf.FixedLenFeature((num_input,), tf.float32),
    'y': tf.FixedLenFeature((num_output,), tf.float32),
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features['X'], parsed_features['y']

# Load Training Data with tf.data
TrainData = tf.data.TFRecordDataset(["../data/TrainingData_robot3_free_robot12_col_"+args.input_type+"_ratio_"+str(args.collision_ratio)+".tfrecord"])
TrainData = TrainData.shuffle(buffer_size=20*batch_size)
TrainData = TrainData.map(parse_proto)
TrainData = TrainData.batch(batch_size)
Trainiterator = TrainData.make_initializable_iterator()
train_batch_x, train_batch_y = Trainiterator.get_next()

# Load Validation Data in Memory
ValidationData = pd.read_csv('../data/ValidationData_robot3_'+args.input_type+'.csv').as_matrix().astype('float32')
X_validation = ValidationData[:,0:num_input]
Y_validation = ValidationData[:,-num_output:]

print('size of train_batch_x: ', train_batch_x.shape)
print('size of train_batch_y: ', train_batch_y.shape)

accuracy_train_all = 0.0
cost_train_all = 0.0

#To Scale wandb Charts
if wandb_use == True:
    wandb_dict = dict()
    wandb_dict['Training Accuracy'] = 0.0
    wandb_dict['Validation Accuracy'] = 0.0
    wandb_dict['Training Cost'] = 1.5
    wandb_dict['Validation Cost'] = 1.5
    wandb.log(wandb_dict)

for epoch in range(training_epochs):
    accuracy_train_local = 0
    accuracy_train_all = 0
    accuracy_val = 0
    reg_train = 0
    reg_val = 0
    cost_train_local = 0
    cost_train_all = 0
    cost_val = 0
    train_batch_num = 0
    
    # Fine Tuning 
    sess.run(Trainiterator.initializer)
    while True:
        try:
            x,y = sess.run([train_batch_x, train_batch_y])
            if (x.shape[0]==batch_size):
                logits_train, hypo_train, _ = sess.run([logits, hypothesis, optimizer], feed_dict={X:x, Y:y,  keep_prob:1.0, is_train:True, learning_rate_tensor:learning_rate})
                train_batch_num = train_batch_num + 1
        except tf.errors.OutOfRangeError:
            break
    # Validation Evaluation
    logits_val, hypo_val = sess.run([logits, hypothesis], feed_dict={X: X_validation, keep_prob: 1.0, is_train:False})
    prediction_val = np.argmax(hypo_val, 1)
    correct_prediction_val = np.equal(prediction_val, np.argmax(Y_validation, 1))
    accuracy_val = np.mean(correct_prediction_val)
    cost_val = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y_validation, logits=logits_val, pos_weight=cross_entropy_weight))

    accuracy_train = accuracy_train_all/train_batch_num
    cost_train = cost_train_all/train_batch_num
    print('Epoch:', '%04d' % (epoch + 1))
    print('Validation Accuracy =', '{:.9f}'.format(accuracy_val))
    print('Validation Cost =', cost_val.eval(session = sess))

    # Log to wandb
    if wandb_use == True:
        wandb_dict = dict()
        wandb_dict['Validation Accuracy'] = accuracy_val
        wandb_dict['Validation Cost'] = cost_val.eval(session = sess)
        wandb.log(wandb_dict)

elapsed_time = time.time() - start_time
print(elapsed_time)
print('Learning Finished!')

# Save Model
saver = tf.train.Saver()
saver.save(sess,'model/model_finetuing_robot12_input5_v5_robot3_free_robot12_col_epoch_'+str(args.training_epoch)+'_ratio_'+str(args.collision_ratio)+'.ckpt')

if wandb_use == True:
    saver.save(sess, os.path.join(wandb.run.dir, 'model/model_finetuing_robot12_input5_v5_robot3_free_robot12_col_epoch_'+str(args.training_epoch)+'_ratio_'+str(args.collision_ratio)+'.ckpt'))
    wandb.config.elapsed_time = elapsed_time

########################### robot3 Test Evaluation ##############################
TestData = pd.read_csv('../data/TestingDataNocut_robot3_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]

logits_test, hypo_test = sess.run([logits, hypothesis], feed_dict={X: X_Test, keep_prob: 1.0, is_train:False})
prediction_test = np.argmax(hypo_test, 1)
correct_prediction_test = np.equal(prediction_test, np.argmax(Y_Test, 1))
accuracy_test = np.mean(correct_prediction_test)
cost_test = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y_Test, logits=logits_test, pos_weight=cross_entropy_weight))

prediction = hypo_test[:, 0]
t = np.arange(0,len(prediction)/test_data_hz, 1/test_data_hz)
print('Test Accuracy(Robot3): ', accuracy_test)
print('Test Cost(Robot3): ', cost_test.eval(session = sess))

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

for i in range(len(prediction)):
    if (Y_Test[i, 0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(prediction[i] >= 0.8):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    if (collision_status == True and JTS_detection == False):
        if(JTS[i] == 1):
            JTS_detection = True
            detection_time_JTS.append(t[i] - collision_time)
    
    if (collision_status == True and DoB_detection == False):
        if(DOB[i] == 1):
            DoB_detection = True
            detection_time_DoB.append(t[i] - collision_time)

    if (Y_Test[i, 0] == 0 and collision_pre == 1):
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
    collision_pre = Y_Test[i, 0]

print('Total collision(Robot3): ', collision_cnt)
print('JTS Failure(Robot3): ', collision_fail_cnt_JTS)
print('NN Failure(Robot3): ', collision_fail_cnt_NN)
print('DOB Failure(Robot3): ', collision_fail_cnt_DoB)
if((collision_cnt - collision_fail_cnt_JTS) != 0):
    print('JTS Detection Time(Robot3): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
if((collision_cnt - collision_fail_cnt_NN) != 0):
    print('NN Detection Time(Robot3): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
if((collision_cnt - collision_fail_cnt_DoB) != 0):
    print('DOB Detection Time(Robot3): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Robot3 Free motion Evaluation 
TestDataFree = pd.read_csv('../data/TestingDataFree_robot3_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
# accu_test, reg_test, cost_test, hypo  = new_saver.get_mean_error_hypothesis(X_TestFree, Y_TestFree)
logits_test, hypo_test = sess.run([logits, hypothesis], feed_dict={X: X_TestFree, keep_prob: 1.0, is_train:False})
prediction_test = np.argmax(hypo_test, 1)
correct_prediction_test = np.equal(prediction_test, np.argmax(Y_TestFree, 1))
accuracy_test = np.mean(correct_prediction_test)
cost_test = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y_TestFree, logits=logits_test, pos_weight=cross_entropy_weight))

# prediction = np.argmax(hypo, 1)
prediction = hypo_test[:, 0]
# prediction = hypo[:, 0]
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(false_positive_local_arr)):
    false_positive_local_arr[j] = (prediction[j]>= 0.8) and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (Robot3): ', sum(false_positive_local_arr))
print('Total Num (Robot3): ', len(Y_TestFree))




############################### robot1 Test Evaluation ######################
TestData = pd.read_csv('../data/TestingDataNocut_robot1_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]
t = np.arange(0,len(prediction)/test_data_hz, 1/test_data_hz)
# accu_test, reg_test, cost_test, hypo  = new_saver.get_mean_error_hypothesis(X_Test, Y_Test)
logits_test, hypo_test = sess.run([logits, hypothesis], feed_dict={X: X_Test, keep_prob: 1.0, is_train:False})
prediction_test = np.argmax(hypo_test, 1)
correct_prediction_test = np.equal(prediction_test, np.argmax(Y_Test, 1))
accuracy_test = np.mean(correct_prediction_test)
cost_test = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y_Test, logits=logits_test, pos_weight=cross_entropy_weight))

# prediction = np.argmax(hypo, 1)
prediction = hypo_test[:, 0]
print('Test Accuracy(Robot1): ', accuracy_test)
print('Test Cost(Robot1): ', cost_test.eval(session = sess))

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

for i in range(len(prediction)):
    if (Y_Test[i, 0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
        JTS_detection = False
        DoB_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(prediction[i] >= 0.8):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    if (collision_status == True and JTS_detection == False):
        if(JTS[i] == 1):
            JTS_detection = True
            detection_time_JTS.append(t[i] - collision_time)
    
    if (collision_status == True and DoB_detection == False):
        if(DOB[i] == 1):
            DoB_detection = True
            detection_time_DoB.append(t[i] - collision_time)

    if (Y_Test[i, 0] == 0 and collision_pre == 1):
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
    collision_pre = Y_Test[i, 0]

print('Total collision(Robot1): ', collision_cnt)
print('JTS Failure(Robot1): ', collision_fail_cnt_JTS)
print('NN Failure(Robot1): ', collision_fail_cnt_NN)
print('DOB Failure(Robot1): ', collision_fail_cnt_DoB)
if((collision_cnt - collision_fail_cnt_JTS) != 0):
    print('JTS Detection Time(Robot1): ', sum(detection_time_JTS)/(collision_cnt - collision_fail_cnt_JTS))
if((collision_cnt - collision_fail_cnt_NN) != 0):
    print('NN Detection Time(Robot1): ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))
if((collision_cnt - collision_fail_cnt_DoB) != 0):
    print('DOB Detection Time(Robot1): ', sum(detection_time_DoB)/(collision_cnt - collision_fail_cnt_DoB))


# Robot1 Free motion Evaluation 
TestDataFree = pd.read_csv('../data/TestingDataFree_robot1_'+args.input_type+'_1000hz.csv').as_matrix().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]
# accu_test, reg_test, cost_test, hypo  = new_saver.get_mean_error_hypothesis(X_TestFree, Y_TestFree)
logits_test, hypo_test = sess.run([logits, hypothesis], feed_dict={X: X_TestFree, keep_prob: 1.0, is_train:False})
prediction_test = np.argmax(hypo_test, 1)
correct_prediction_test = np.equal(prediction_test, np.argmax(Y_TestFree, 1))
accuracy_test = np.mean(correct_prediction_test)
cost_test = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y_TestFree, logits=logits_test, pos_weight=cross_entropy_weight))

prediction = hypo_test[:, 0]
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(false_positive_local_arr)):
    false_positive_local_arr[j] = (prediction[j]>= 0.8) and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive (Robot1): ', sum(false_positive_local_arr))
print('Total Num (Robot1): ', len(Y_TestFree))
