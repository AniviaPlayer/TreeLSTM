#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: Learner.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
import numpy as np
from TreeLSTM import TreeLSTM
from TreeMgr import TreeMgr

MEM_DIM = 200
IN_DIM = 400
SENTENCE_FILE=""
DESCRIPT_FILE=""

class Learner(object):
    """
    must add some parsing function
    """
    def __init__(self,sentence_file,descript_file):
        self.mgr = TreeMgr(sentence_file,descript_file)
        self.num_training_samples = mgr.get_num_sample()
        self.rng = np.random.RandomState(123)
        self.treelstm = TreeLSTM(mgr,rng,MEM_DIM,IN_DIM)

    def batch_train(self,training_iters=100,lr=0.01):
        print("Batch Training")
        self.mini_batch_train(self.num_training_samples)

    def mini_batch_train(self,batch_size=25,training_iters=100,lr=0.01):
        for i in range(training_iters):
            cost = 0.0
            sentence_iterator = self.mgr.sentence_iterator()
            batch_num = 1
            print("======")
            print("Training Iteration {}:".format(i))
            print("======")
            for i ,sentence in enumerate(sentence_iterator):
                cost += self.treelstm.forward_pass(sentence,labelMap[sentence])
                if (i+1) % batch_size == 0:
                    gparams = [T.grad(cost,param) for param in self.treelstm.params]
                    updates = [(param,param - lr * gparam ) \
                            for (param,gparam) in zip(self.treelstm.params,gparams)]
                    for e in updates:
                        tmp_new = e[1].eval({})
                        e[0].set_value(tmp_new)
                    print("Batch {} cost: {}"..format(batch_num,cost.eval({})))
                    batch += 1
                    cost = 0.0
def main():
    learner = Learner(SENTENCE_FILE,DESCRIPT_FILE)
    learner.batch_train()

if __name__ = "__main__":
    main()
