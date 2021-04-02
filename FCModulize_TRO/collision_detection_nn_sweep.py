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

args_wandb = dict(
    batch_size = 100,
    hidden_neuron = 15,
    cross_entropy_weight = 1.0
    )

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
parser.add_argument('--use_wandb', type=str2bool, default=True) # Use Logging tool
parser.add_argument('--use_gpu', type=str2bool, default=False) # Use GPU
parser.add_argument('--use_narrow_structure', type=str2bool, default=False) # Use half hidden neuron as layer goes deeper
parser.add_argument('--use_ee_acc_data', type=str2bool, default=False) # Use end effector acceleration data
parser.add_argument('--use_tf_record', type=str2bool, default=False) # Use tf record format for large data
parser.add_argument('--learning_rate', type=float, default=0.00001) # Learning rate
parser.add_argument('--training_epoch', type=int, default=10) # Training epoch
parser.add_argument('--batch_size', type=int, default=1000) # Size of batch
parser.add_argument('--drop_out_rate', type=float, default=0.0) # Drop out rate
parser.add_argument('--regularization_factor', type=float, default=0.0000001) # Regularization
parser.add_argument('--hidden_neuron', type=int, default=15) # Number of hidden neuron
parser.add_argument('--cross_entropy_weight', type=float, default=1.0) # Weighted cross entropy 
parser.add_argument('--test_every', type=int, default=20) # Print test result every test_every epoch
parser.add_argument('--test_threshold', type=float, default=0.5) # Weighted cross entropy 
args = parser.parse_args()

# Init wandb
wandb_use = args.use_wandb  
if wandb_use == True:
    wandb.init(project="TRO", tensorboard=False, config= args_wandb)

# Number of Input/Output Data
time_step = 5
num_data_type = 6
num_one_joint_data = time_step * (num_data_type-1)
num_joint = 6
if args.use_ee_acc_data is False :
    num_input = num_one_joint_data*num_joint # joint data
    num_concatenate_node = 6
else:
    num_input = num_one_joint_data*num_joint + time_step # joint data + ee_acc data
    num_concatenate_node = 7
num_output = 2

# Hyper parameter Setting
learning_rate = args.learning_rate
training_epochs = args.training_epoch
batch_size = args.batch_size
drop_out_rate = args.drop_out_rate
regul_factor = args.regularization_factor
cross_entropy_weight = args.cross_entropy_weight
hidden_neurons = args.hidden_neuron
hidden_neurons_2nd = hidden_neurons
hidden_neurons_3rd = hidden_neurons
test_every = args.test_every
test_threshold = args.test_threshold
if args.use_narrow_structure is True:
    hidden_neurons_2nd = math.ceil(hidden_neurons/2)
    hidden_neurons_3rd = math.ceil(hidden_neurons/4)


# Tensorflow Setting
if args.use_gpu is False :
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=16,
        intra_op_parallelism_threads=16, log_device_placement=False)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = False
    sess = tf.compat.v1.Session(config=tf_config)
else :
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sess = tf.compat.v1.Session()

# Neural Network
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        

    def _build_net(self):
        with tf.compat.v1.variable_scope(self.name, reuse = tf.compat.v1.AUTO_REUSE):
            tf.compat.v1.disable_eager_execution()
            self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, num_input], name = "input")
            self.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_output], name= "output")
            self.is_train = tf.compat.v1.placeholder(tf.bool, name = "is_train")
            self.drop_out_rate = tf.compat.v1.placeholder(tf.float32, name="drop_out_rate")
            self.cross_entropy_weight = tf.compat.v1.placeholder(tf.float32, name="cross_entropy_weight")
            self.hidden_layers = 0
            self.hidden_neurons = hidden_neurons

            # Joint Data Layers
            for i in range(num_joint):
                with tf.compat.v1.variable_scope("Joint"+str(i)+"Net"):
                    W1 = tf.compat.v1.get_variable("W1", shape=[num_one_joint_data, self.hidden_neurons], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                    b1 = tf.Variable(tf.compat.v1.random_normal([self.hidden_neurons]))
                    L1 = tf.matmul(self.X[:, num_one_joint_data*i:num_one_joint_data*(i+1)], W1) +b1
                    L1 = tf.compat.v1.layers.batch_normalization(L1, training=self.is_train)
                    L1 = tf.nn.relu(L1)
                    L1 = tf.nn.dropout(L1, rate=self.drop_out_rate)

                    W2 = tf.compat.v1.get_variable("W2", shape=[self.hidden_neurons, hidden_neurons_2nd], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                    b2 = tf.Variable(tf.compat.v1.random_normal([hidden_neurons_2nd]))
                    L2 = tf.matmul(L1, W2) +b2
                    L2 = tf.compat.v1.layers.batch_normalization(L2, training=self.is_train)
                    L2 = tf.nn.relu(L2)
                    L2 = tf.nn.dropout(L2, rate=self.drop_out_rate)
                    self.hidden_layers += 1

                    W3 = tf.compat.v1.get_variable("W3", shape=[hidden_neurons_2nd, hidden_neurons_3rd], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                    b3 = tf.Variable(tf.compat.v1.random_normal([hidden_neurons_3rd]))
                    L3 = tf.matmul(L2, W3) +b3
                    L3 = tf.compat.v1.layers.batch_normalization(L3, training=self.is_train)
                    L3 = tf.nn.relu(L3)
                    L3 = tf.nn.dropout(L3, rate=self.drop_out_rate)
                    self.hidden_layers += 1

                    W4 = tf.compat.v1.get_variable("W4", shape=[hidden_neurons_3rd, 1], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                    b4 = tf.Variable(tf.compat.v1.random_normal([1]))
                    L4 = tf.matmul(L3, W4) +b4
                    L4 = tf.compat.v1.layers.batch_normalization(L4, training=self.is_train)
                    L4 = tf.nn.relu(L4)
                    L4 = tf.nn.dropout(L4, rate=self.drop_out_rate)
                    if(i == 0):
                        self.LConcat = L4
                    else:
                        self.LConcat = tf.concat([self.LConcat, L4],1)
                     
            # End Effector Accerlation Data Layers   
            if args.use_ee_acc_data is True :
                W_ee1 = tf.compat.v1.get_variable("W_ee1", shape=[time_step, self.hidden_neurons], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b_ee1 = tf.Variable(tf.compat.v1.random_normal([self.hidden_neurons]))
                L_ee1 = tf.matmul(self.X[:, num_one_joint_data*6:num_one_joint_data*6+time_step], W_ee1) + b_ee1
                L_ee1 = tf.compat.v1.layers.batch_normalization(L_ee1, training=self.is_train)
                L_ee1 = tf.nn.relu(L_ee1)
                L_ee1 = tf.nn.dropout(L_ee1, rate=self.drop_out_rate)

                W_ee2 = tf.compat.v1.get_variable("W_ee2", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b_ee2 = tf.Variable(tf.compat.v1.random_normal([self.hidden_neurons]))
                L_ee2 = tf.matmul(L_ee1, W_ee2) + b_ee2
                L_ee2 = tf.compat.v1.layers.batch_normalization(L_ee2, training=self.is_train)
                L_ee2 = tf.nn.relu(L_ee2)
                L_ee2 = tf.nn.dropout(L_ee2, rate=self.drop_out_rate)
                self.hidden_layers += 1
                
                W_ee3 = tf.compat.v1.get_variable("W_ee3", shape=[self.hidden_neurons, 1], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b_ee3 = tf.Variable(tf.compat.v1.random_normal([1]))
                L_ee3 = tf.matmul(L_ee2, W_ee3) +b_ee3
                L_ee3 = tf.compat.v1.layers.batch_normalization(L_ee3, training=self.is_train)
                L_ee3 = tf.nn.relu(L_ee3)
                L_ee3 = tf.nn.dropout(L_ee3, rate=self.drop_out_rate)
                self.hidden_layers += 1
                self.LConcat = tf.concat([self.LConcat, L_ee3],1)

            # Concatenating joint nodes and end effector nodes
            with tf.compat.v1.variable_scope("ConcatenateNet"):
                W5 = tf.compat.v1.get_variable("W5", shape=[num_concatenate_node, self.hidden_neurons], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b5 = tf.Variable(tf.compat.v1.random_normal([self.hidden_neurons]))
                L5 = tf.matmul(self.LConcat, W5) +b5
                L5 = tf.compat.v1.layers.batch_normalization(L5, training=self.is_train)
                L5 = tf.nn.relu(L5)
                L5 = tf.nn.dropout(L5, rate=self.drop_out_rate)
                self.hidden_layers += 1

                W6 = tf.compat.v1.get_variable("W6", shape=[self.hidden_neurons, self.hidden_neurons], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b6 = tf.Variable(tf.compat.v1.random_normal([self.hidden_neurons]))
                L6 = tf.matmul(L5, W6) +b6
                L6 = tf.compat.v1.layers.batch_normalization(L6, training=self.is_train)
                L6 = tf.nn.relu(L6)
                L6 = tf.nn.dropout(L6, rate=self.drop_out_rate)
                self.hidden_layers += 1

                W7 = tf.compat.v1.get_variable("W7", shape=[self.hidden_neurons, num_output], initializer=tf.keras.initializers.glorot_uniform, regularizer=tf.keras.regularizers.L2(l2=regul_factor))
                b7 = tf.Variable(tf.compat.v1.random_normal([num_output]))
                self.logits = tf.matmul(L6, W7) + b7
                tf.identity(self.logits, "logits")
                self.hypothesis = tf.nn.softmax(self.logits)
                self.hypothesis = tf.identity(self.hypothesis, "hypothesis")

            # define cost/loss & optimizer
            self.l2_reg = sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=self.Y, logits=self.logits, pos_weight=self.cross_entropy_weight))

            self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost + self.l2_reg)
        
        self.prediction = tf.argmax(self.hypothesis, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def get_mean_error_hypothesis(self, x_test, y_test, drop_out_rate=0.0, is_train=False, cross_entropy_weight=1.0):
        return self.sess.run([self.accuracy,  self.l2_reg, self.cost, self.hypothesis], feed_dict={self.X: x_test, self.Y: y_test, self.drop_out_rate: drop_out_rate, self.is_train: is_train, self.cross_entropy_weight: cross_entropy_weight})

    def train(self, x_data, y_data, drop_out_rate=0.0, is_train=True, cross_entropy_weight=1.0):
        return self.sess.run([self.accuracy, self.l2_reg, self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.drop_out_rate: drop_out_rate, self.is_train: is_train, self.cross_entropy_weight: cross_entropy_weight})

    def get_hidden_number(self):
        return [self.hidden_layers, self.hidden_neurons]


m1 = Model(sess, "m1")
sess.run(tf.compat.v1.global_variables_initializer())

# Log Configuration
if wandb_use == True:
    wandb.config.epoch = training_epochs
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = learning_rate
    wandb.config.drop_out_rate = drop_out_rate
    wandb.config.num_input = num_input
    wandb.config.num_output = num_output
    wandb.config.time_step = time_step
    wandb.config.hidden_layers, wandb.config.hidden_neurons = m1.get_hidden_number()
    wandb.config.L2_regularization = regul_factor
    wandb.config.cross_entropy_weight = cross_entropy_weight
    wandb.config.use_narrow_structure = args.use_narrow_structure

# When Using tfrecord
if args.use_tf_record is True:
    # Load Training Data with tf.data
    def parse_proto(example_proto):
        features = {
            'X': tf.compat.v1.FixedLenFeature((num_input,), tf.float32),
            'y': tf.compat.v1.FixedLenFeature((num_output,), tf.float32),
        }
        parsed_features = tf.compat.v1.parse_single_example(example_proto, features)
        return parsed_features['X'], parsed_features['y']
    TrainData = tf.data.TFRecordDataset(["../data_tro/TrainingData4Cut.tfrecord"])
    TrainData = TrainData.shuffle(buffer_size=100*batch_size)
    TrainData = TrainData.map(parse_proto)
else:
    # Load Training Data in Memory
    TrainDataRaw = pd.read_parquet('../data_tro/TrainingData4.parquet').to_numpy().astype('float32')
    TrainData = tf.data.Dataset.from_tensor_slices((TrainDataRaw[:,0:num_input], TrainDataRaw[:,-num_output:]))
    TrainData = TrainData.shuffle(buffer_size=100*batch_size)
TrainData = TrainData.batch(batch_size)
TrainData = TrainData.prefetch(buffer_size=1)
Trainiterator = tf.compat.v1.data.make_initializable_iterator(TrainData)
train_batch_x, train_batch_y = Trainiterator.get_next()

# Load Validation Data in Memory
ValidationData = pd.read_csv('../data_tro/ValidationData4.csv').to_numpy().astype('float32')
X_validation = ValidationData[:,0:num_input]
Y_validation = ValidationData[:,-num_output:]

# Train Model
# Logging: To Scale wandb Charts
if wandb_use == True:
    wandb_dict = dict()
    wandb_dict['Training Accuracy'] = 0.0
    wandb_dict['Validation Accuracy'] = 0.0
    wandb_dict['Training Cost'] = 1.5
    wandb_dict['Validation Cost'] = 1.5
    wandb.log(wandb_dict)

test_result = []
# Training
for epoch in range(training_epochs):
    accu_train = 0
    accu_val = 0
    reg_train = 0
    reg_val = 0
    cost_train = 0
    cost_val = 0
    train_batch_num = 0
    validation_batch_num = 0

    print('-----------------------------------------------------------------')
    print('Epoch:', '%04d' % (epoch + 1))
    # Training Data
    sess.run(Trainiterator.initializer)
    while True:
        try:
            x,y = sess.run([train_batch_x, train_batch_y])
            if (x.shape[0]==batch_size):
                accu, reg_c, cost,_ = m1.train(x, y, drop_out_rate=drop_out_rate, is_train=True, cross_entropy_weight=cross_entropy_weight)
                train_batch_num = train_batch_num + 1
                accu_train = ((train_batch_num-1)*accu_train + accu )/ train_batch_num
                reg_train = ((train_batch_num-1)*reg_train + reg_c )/ train_batch_num
                cost_train = ((train_batch_num-1)*cost_train + cost )/ train_batch_num
        except tf.errors.OutOfRangeError:
            break
    # Validation Evaluation
    accu_val, reg_val, cost_val, _ = m1.get_mean_error_hypothesis(X_validation, Y_validation)
    print('Train Accuracy =', '{:.9f}'.format(accu_train))
    print('Validation Accuracy =', '{:.9f}'.format(accu_val))
    print('Train Cost =', '{:.9f}'.format(cost_train), 'Train Regul =', '{:.9f}'.format(reg_train))
    print('Validation Cost =', '{:.9f}'.format(cost_val), 'Validation Regul =', '{:.9f}'.format(reg_val))

    # Log to wandb
    if wandb_use == True:
        wandb_dict = dict()
        wandb_dict['Training Accuracy'] = accu_train
        wandb_dict['Validation Accuracy'] = accu_val
        wandb_dict['Training Cost'] = cost_train
        wandb_dict['Training Reg'] = reg_train
        wandb_dict['Validation Cost'] = cost_val
        wandb_dict['Validation Reg'] = reg_val
        wandb.log(wandb_dict)

# Load Test Data
TestData = pd.read_parquet('../data_tro/TestingDataCollision4.parquet').to_numpy().astype('float32')
X_Test = TestData[:,0:num_input]
Y_Test = TestData[:,-num_output:]
JTS = TestData[:,num_input]
DOB = TestData[:,num_input+1]

# Load Test Free Data
TestDataFree = pd.read_parquet('../data_tro/TestingDataFree4.parquet').to_numpy().astype('float32')
X_TestFree = TestDataFree[:,0:num_input]
Y_TestFree = TestDataFree[:,-num_output:]

# Test Evaluation
accu_test, reg_test, cost_test, hypo  = m1.get_mean_error_hypothesis(X_Test, Y_Test)
prediction = np.argmax(hypo, 1)
t = np.arange(0,0.001*len(prediction),0.001)
print('------------------Test--------------------')
print('Test Accuracy: ', accu_test)
print('Test Cost: ', cost_test)

collision_pre = 0
collision_cnt = 0
collision_time = 0
detection_time_NN = []
collision_status = False
NN_detection = False
collision_fail_cnt_NN = 0

for i in range(len(prediction)):
    if (Y_Test[i,0] == 1 and collision_pre == 0):
        collision_cnt = collision_cnt +1
        collision_time = t[i]
        collision_status = True
        NN_detection = False
    
    if (collision_status == True and NN_detection == False):
        if(hypo[i,0] > test_threshold):
            NN_detection = True
            detection_time_NN.append(t[i] - collision_time)

    if (Y_Test[i,0] == 0 and collision_pre == 1):
        collision_status = False
        if(NN_detection == False):
            detection_time_NN.append(0.0)
            collision_fail_cnt_NN = collision_fail_cnt_NN+1
    collision_pre = Y_Test[i,0]

print('Total collision: ', collision_cnt)
print('NN Failure: ', collision_fail_cnt_NN)
print('NN Detection Time: ', sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN))

# Free motion Evaluation
accu_test, reg_test, cost_test, hypofree  = m1.get_mean_error_hypothesis(X_TestFree, Y_TestFree)
false_positive_local_arr = np.zeros((len(Y_TestFree),1))
for j in range(len(false_positive_local_arr)):
    false_positive_local_arr[j] = hypofree[j,0] > test_threshold and np.equal(np.argmax(Y_TestFree[j,:]), 1)
print('False Positive: ', sum(false_positive_local_arr))
print('Total Num: ', len(Y_TestFree))
test_result.append([sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN), sum(false_positive_local_arr)])

# Log to wandb
if wandb_use == True:
    wandb_dict = dict()
    wandb_dict['NN Detection Time'] = sum(detection_time_NN)/(collision_cnt - collision_fail_cnt_NN)
    wandb_dict['False Positive'] = sum(false_positive_local_arr)
    wandb.log(wandb_dict)

elapsed_time = time.time() - start_time
print(elapsed_time)
print('Learning Finished!')