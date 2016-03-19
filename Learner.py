#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: Learner.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
import numpy as np
from util import *
from collections import Counter
from TreeMgr import TreeMgr
from TreeLSTM import TreeLSTM

MEM_DIM = 200
IN_DIM = 300
SENTENCE_FILE="laptop/sents.txt"
DESCRIPT_FILE="laptop/cparents.txt"
LABEL_FILE = "laptop/labels"


class Learner(object):
    """
    must add some parsing function
    """
    def __init__(self,sentence_file,descript_file,label_file,param_file=None):
        self.mgr = TreeMgr(sentence_file,descript_file)
        self.num_samples = self.mgr.get_num_sample()
        self.rng = np.random.RandomState(123)
        self.treelstm = TreeLSTM(self.mgr,self.rng,MEM_DIM,IN_DIM,param_file)
        self.label_list = io.loadLabel(label_file)

    def batch_train(self,training_iters=100,lr=0.01):
        print("Batch Training")
        self.mini_batch_train(self.num_samples)

    def mini_batch_train(self,batch_size=25,training_iters=100,lr=1,rho=1e-3):
        for i in range(training_iters):
            batch_cost = 0.0
            sentence_iterator = self.mgr.sentence_iterator()
            batch_num = 1
            print("======")
            print("Training Iteration {}:".format(i))
            print("======")
            correct_list = []
            correct = 0
            pred_list = []
            for i ,sentence in enumerate(sentence_iterator):
                pred,cost,prob = self.treelstm.forward_pass(sentence,int(self.label_list[i]))
                if pred == int(self.label_list[i]):
                    correct += 1
                batch_cost += cost
                print(prob)
                correct_list.append(int(self.label_list[i]))
                pred_list.append(pred)
                self.treelstm.back_prop(prob,sentence,int(self.label_list[i]))
                if (i+1) % batch_size == 0:
                    batch_cost += rho * np.sum([np.sum(param**2) for param in self.treelstm.params]) #L2
                    print("GOLDEN : {}".format(Counter(correct_list)))
                    print("PRED : {}".format(Counter(pred_list)))
                    print("Batch {} cost: {} accuracy: {:.3f}%".format(batch_num,batch_cost,float(correct)*100/batch_size))
                    self.treelstm.update()
                    print(self.treelstm.params[1])
                    batch_num += 1
                    batch_cost = 0.0
                    correct = 0
                    correct_list = []
                    pred_list = []
    # def validate(self):
        # """
        # param_file in constructor must be assigned!
        # """
        # print("Validating...")
        # pred_list = []
        # correct_num = 0
        # sentence_iterator = self.mgr.sentence_iterator()
        # for i ,sentence in enumerate(sentence_iterator):
            # _,pred = self.treelstm.forward_pass(sentence,self.label_list[i]).sum()
            # pred_list.append(pred)
            # if np.argmax(pred) == label_list[i]:
                # correct_num += 1
        # print("Accuracy {:.3f} ({}/{})".format(float(correct_num/self.num_samples),correct_num,self.num_samples))
        # print(Counter(label_list))
        # print(Counter(pred_list))
def main():
    learner = Learner(SENTENCE_FILE,DESCRIPT_FILE,LABEL_FILE)
    learner.mini_batch_train(100)
    # validationer = Learner(DEV_SENTENCE_FILE,DEV_DESCRIPT_FILE,DEV_LABEL_FILE)
    # validationer.validate()

if __name__ == "__main__":
    main()
