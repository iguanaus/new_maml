""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images
import pickle

FLAGS = flags.FLAGS
#random.seed(123490234)

#This method will create all the data.
#task_id = "C-sin"
# task_id = "C-sin"
# task_id = "bounce-states"
#num_shots = 10
#dataset_PATH = "data/"
#filename = dataset_PATH + task_id + "_{0}-shot_2.p".format(num_shots)
#tasks = pickle.load(open(filename, "rb"))

filename = "data/C-sin_10-shot_legit_2.p"
#batch_size = 25

tasks = pickle.load(open(filename, "rb"))

def convertData(batch_size,myTrain):
    num_batches = len(myTrain)/batch_size
    print("My Train length: " , len(myTrain), " batch size: " , batch_size, " num batches: " , num_batches)
    #Now pick each one of the groups
    allTrainData = []
    for i in xrange(0,num_batches):
        tasks_for_batch = myTrain[i*batch_size:(i+1)*batch_size]
        inputAll = np.array([])
        labelAll = np.array([])
        for task in tasks_for_batch:
            data = task[0]
            inputa = data[0][0]
            labela = data[0][1]
            inputb = data[1][0]
            labelb = data[1][1]           
            inputs = np.vstack((inputa,inputb)).reshape(1,-1,1)
            labels = np.vstack((labela,labelb)).reshape(1,-1,1)
            if inputAll.size == 0:
                inputAll = inputs
                labelAll = labels
            else:
                inputAll = np.vstack((inputAll,inputs))
                labelAll = np.vstack((labelAll,labels))
        allTrainData.append([inputAll,labelAll,0,0])
    return allTrainData


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.allTrainData = None
        self.allTestData = None
        self.iterCount = 0

        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1

    def setupData(self):
        oldVersion = False
        if oldVersion:
            num_tasks = FLAGS.limit_task_num #Number of train tasks
            numTestBatches = FLAGS.numTestBatches #Number of test tasks
            numTest=100 #How big to make the test batch size.
            print("setupData. setting up data.....")
            self.allTrainData = []
            self.allTestData = []
            #This is how many unique task to make.
            for i in xrange(0,num_tasks):
                self.allTrainData.append(self.generate_sinusoid_batch(usePreValues=False))
            ordd = self.batch_size
            self.batch_size = numTest
            for i in xrange(0,numTestBatches):
                self.allTestData.append(self.generate_sinusoid_batch(usePreValues=False))
            self.batch_size = ordd
            print("setupData. Done setting up data....")
        else:
            self.allTrainData = convertData(self.batch_size,tasks['tasks_train'])
            self.allTestData = convertData(FLAGS.test_batch_amount,tasks['tasks_test'])
            print(len(self.allTrainData))
            print(len(self.allTestData))
        print("Done with setup....")

    def getPreData(self,num_tasks=100,train=True,numTestBatches=1):
        if train:
            idRet = self.iterCount
            self.iterCount += 1
            if (self.iterCount > (len(self.allTrainData)-1)):
                self.iterCount = 0
            print("Id return: " , idRet)
            return self.allTrainData[idRet]
        else:
            if numTestBatches > 1:
                numTestBatches = len(self.allTestData)
            idRet = self.iterCount
            self.iterCount += 1
            if (self.iterCount > (numTestBatches-1)):
                self.iterCount = 0
            print("testing..: " , ranId)
            return self.allTestData[ranId]

    def generate_sinusoid_batch(self, train=True, input_idx=None,usePreValues=True,numTotal=None):
        numTestBatches = FLAGS.numTestBatches
        if numTotal == None:
            numTotal = FLAGS.limit_task_num

        if FLAGS.limit_task == True and usePreValues:
            #print("us")
            if self.allTrainData == None:
                self.setupData()
            return self.getPreData(num_tasks=numTotal,train=train,numTestBatches=numTestBatches)

        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase
