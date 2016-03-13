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
import theano
import theano.tensor as T
from collections import Counter
from TreeMgr import TreeMgr
from TreeLSTM import TreeLSTM

MEM_DIM = 200
IN_DIM = 300
SENTENCE_FILE="laptop/sents.txt"
DESCRIPT_FILE="laptop/cparents.txt"


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

    def mini_batch_train(self,batch_size=25,training_iters=100,lr=0.01):
        for i in range(training_iters):
            cost = 0.0
            sentence_iterator = self.mgr.sentence_iterator()
            batch_num = 1
            print("======")
            print("Training Iteration {}:".format(i))
            print("======")
            for i ,sentence in enumerate(sentence_iterator):
                cost,_ += self.treelstm.forward_pass(sentence,self.label_list[i]).sum()
                if (i+1) % batch_size == 0:
                    gparams = [T.grad(cost,param) for param in self.treelstm.params]
                    updates = [(param,param - lr * gparam ) \
                            for (param,gparam) in zip(self.treelstm.params,gparams)]
                    for e in updates:
                        tmp_new = e[1].eval({})
                        e[0].set_value(tmp_new)
                    print("Batch {} cost: {}".format(batch_num,cost.eval({})))
                    batch_num += 1
                    cost = 0.0
    def validate(self):
        """
        param_file in constructor must be assigned!
        """
        print("Validating...")
        pred_list = []
        correct_num = 0
        sentence_iterator = self.mgr.sentence_iterator()
        for i ,sentence in enumerate(sentence_iterator):
            _,pred = self.treelstm.forward_pass(sentence,self.label_list[i]).sum()
            pred_list.append(pred)
            if np.argmax(pred) == label_list[i]:
                correct_num += 1
        print("Accuracy {:.3f} ({}/{})".format(float(correct_num/self.num_samples),correct_num,self.num_samples))
        print(Counter(label_list))
        print(Counter(pred_list))
def main():
    learner = Learner(SENTENCE_FILE,DESCRIPT_FILE,LABEL_FILE)
    learner.mini_batch_train(100)
    validationer = Learner(DEV_SENTENCE_FILE,DEV_DESCRIPT_FILE,DEV_LABEL_FILE)
    validationer.validate()

if __name__ == "__main__":
    main()
