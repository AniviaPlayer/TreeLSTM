#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: treelstm.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: TreeLSTM
"""
import numpy as np
import pickle

def sigmoid(x):
    """
    x: a numpy array
    """
    return 1 / (1 + np.exp(-x))

def softmax(w):
    """
    w: a numpy array
    """
    e = np.exp(w)
    return (e / np.sum(e))

def cross_entropy(golden,pred):
    """
    golden: a numpy array (one hot)
    pred: a numpy array
    """
    return -1.0 * (pred*np.log(golden)).sum()

class TreeLSTM(object):

    def __init__(self,mgr,rng,mem_dim,in_dim,param_file=None,
            num_classes=3,inner_activation=sigmoid,outer_activation=np.tanh,lamda=1e-3):
        """
        mgr: TreeMgr
        rng : random state
        param_file : param file (json format)
        activation : non-linear function
        """
        self.mgr = mgr
        self.rng = rng
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.inner_activation = inner_activation
        self.outer_activation = outer_activation
        self.lamda = lamda

        self.init_params()
        if param_file is not None:
            self.load_params(param_file)

        print("Modle Initialization finish!")

    def init_params(self):
        self.bi = np.zeros(shape=(self.mem_dim,))
        self.bf = np.zeros(shape=(self.mem_dim,))
        self.bo = np.zeros(shape=(self.mem_dim,))
        self.bu = np.zeros(shape=(self.mem_dim,))
        self.bs = np.zeros(shape=(self.num_classes,))
        self.Wi = self.rng.uniform(size=(self.mem_dim,self.in_dim)) * 0.5
        self.Wo = self.rng.uniform(size=(self.mem_dim,self.in_dim)) * 0.5
        self.Ws = self.rng.uniform(size=(self.num_classes,self.mem_dim)) * 0.01
        self.Ui = self.rng.uniform(size=(self.mem_dim,self.mem_dim * 2)) * 0.05
        self.Uf = self.rng.uniform(size=(self.mem_dim,self.mem_dim * 2)) * 0.05
        self.Uo = self.rng.uniform(size=(self.mem_dim,self.mem_dim * 2)) * 0.05
        self.Uu = self.rng.uniform(size=(self.mem_dim,self.mem_dim * 2)) * 0.01

        self.params = [self.Wi,self.Wo,self.Ws,
                      self.bi,self.bf,self.bo,self.bu,self.bs,
                      self.Ui,self.Uf,self.Uo,self.Uu]
        self.dWi = np.zeros(self.Wi.shape)
        self.dWs = np.zeros(self.Ws.shape)
        self.dWo = np.zeros(self.Wo.shape)
        self.dbi = np.zeros(self.bi.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dbo = np.zeros(self.bo.shape)
        self.dbu = np.zeros(self.bu.shape)
        self.dbs = np.zeros(self.bs.shape)
        self.dUi = np.zeros(self.Ui.shape)
        self.dUf = np.zeros(self.Uf.shape)
        self.dUo = np.zeros(self.Uo.shape)
        self.dUu = np.zeros(self.Uu.shape)

        self.grads = [self.dWi,self.dWo,self.dWs,
                    self.dbi,self.dbf,self.dbo,self.dbu,self.dbs,
                    self.dUi,self.dUf,self.dUo,self.dUu]
    ##########
    #  Gate  #
    ##########
    def composer_i(self,h):
        return self.inner_activation(np.dot(self.Ui,h) + self.bi)

    def composer_f(self,h):
        return self.inner_activation(np.dot(self.Uf,h) + self.bf)

    def composer_o(self,h):
        return self.inner_activation(np.dot(self.Uo,h) + self.bo)

    def composer_u(self,h):
        return self.outer_activation(np.dot(self.Uu,h) + self.bu)

    def leaf_i(self,x):
        return self.inner_activation(np.dot(self.Wi,x) + self.bi)

    def leaf_o(self,x):
        return self.inner_activation(np.dot(self.Wo,x) + self.bo)

    def combine_c(self,cr,cl):  #can define combine method
        return cr + cl

    def output_root(self,h):
        return softmax(np.dot(self.Ws,h) + self.bs)

    #########
    #  I/O  #
    #########
    def load_params(self,param_file):
        self.params = pickle.load(param_file)

    def save_params(self,param_file):
        pickle.dump(self.params)


    def forward_pass(self,sentence,label):
        inpt_tree = self.mgr.get_tree(sentence)
        golden = label
        one_hot_golden = np.ones(shape=(self.num_classes,1))*1e-9
        one_hot_golden[golden] = 1
        cost = self.forward_pass_node(inpt_tree.root)
        pred = self.output_root(inpt_tree.root.h)
        cost  = cross_entropy(one_hot_golden,pred)
        return np.argmax(pred),cost,pred

    def forward_pass_node(self,node,test=False):
        cost = 0.0
        if node.is_leaf():
            x = self.mgr.get_glove_vec(node.word)
            node.c = self.leaf_i(x)
            node.o = self.leaf_o(x)
            node.h = node.o * self.outer_activation(node.c)
        else:
            cost_l = self.forward_pass_node(node.l_child,test)
            cost_r = self.forward_pass_node(node.r_child,test)
            cost += cost_l + cost_r
            l_child,r_child = node.get_child()
            children = np.hstack((l_child.h,r_child.h))
            node.i = self.composer_i(children)
            node.f = self.composer_f(children)
            node.o = self.composer_o(children)
            node.u = self.composer_u(children)
            node.c = node.i * node.u + node.f * self.combine_c(l_child.c,r_child.c)
            node.h = node.o * self.outer_activation(node.c)
        return cost

    def back_prop(self,pred,sentence,label):
        inpt_tree = self.mgr.get_tree(sentence)
        deltas = pred[int(label)] - 1.0
        self.dWs += np.outer(deltas,inpt_tree.root.h)
        self.dbs += deltas
        self.back_prop_node(inpt_tree.root,deltas)

    def back_prop_node(self,node,epsH,epsC=None):
        epsO = epsH * self.outer_activation(node.c) * node.o * (1-node.o)
        if epsC is None:
            epsC  = epsH * node.o * (1-(self.outer_activation(node.c) ** 2))
        else:
            epsC += epsH * node.o * (1-(self.outer_activation(node.c) ** 2))

        if node.is_leaf():
            x = self.mgr.get_glove_vec(node.word)
            self.dWo += np.outer(epsO,x)
            self.dbo += epsO
            self.dWi += np.outer(epsC,x)
            self.dbi += epsC
        else:
            l_child,r_child = node.get_child()
            childs_h = np.hstack((l_child.h,r_child.h))
            self.dUo += np.outer(epsO,childs_h)
            self.dbo += epsO


            epsI = epsC * node.u * node.i * (1-node.i)
            self.dbi += epsI
            self.dUi += np.outer(epsI,childs_h)

            epsU = epsC * node.i * (1-(self.outer_activation(node.u) ** 2))
            self.dbu += epsU
            self.dUu += np.outer(epsU,childs_h)

            epsF = epsC * self.combine_c(l_child.c,r_child.c) * node.f * (1 - node.f)
            self.dbf = epsF
            self.dUf = np.outer(epsF,childs_h)

            epsH = epsO.dot(self.Uo) + epsI.dot(self.Ui) + epsU.dot(self.Uu) + epsF.dot(self.Uf)
            epsC *= node.f
            self.back_prop_node(l_child,epsH[:self.mem_dim],epsC)
            self.back_prop_node(r_child,epsH[self.mem_dim:],epsC)

    def update(self):
        self.params = [P + 1e-2 * dP for P,dP in zip(self.params,self.grads)]
        self.resetgrads()

    def resetgrads(self):
        self.dWi = np.zeros(self.Wi.shape)
        self.dWs = np.zeros(self.Ws.shape)
        self.dWo = np.zeros(self.Wo.shape)
        self.dbi = np.zeros(self.bi.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dbo = np.zeros(self.bo.shape)
        self.dbu = np.zeros(self.bu.shape)
        self.dbs = np.zeros(self.bs.shape)
        self.dUi = np.zeros(self.Ui.shape)
        self.dUf = np.zeros(self.Uf.shape)
        self.dUo = np.zeros(self.Uo.shape)
        self.dUu = np.zeros(self.Uu.shape)

if __name__ == "__main__":
    pass
