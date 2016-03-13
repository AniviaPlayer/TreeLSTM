#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: treelstm.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: TreeLSTM
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax,sigmoid,categorical_crossentropy


class TreeLSTM(object):

    def __init__(self,mgr,rng,mem_dim,in_dim,num_classes=3,inner_activation=sigmoid,outer_activation=sigmoid,param_file=None):
        """
        mgr
        rng : random state
        param_file : param file
        activation : non-linear function
        """
        self.mgr=mgr
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.inner_activation = inner_activation
        self.outer_activation = outer_activation
        ##Setting params in treeLSTM
        if param_file == None:
            self.bi = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,)),dtype=theano.config.floatX),name="bi",borrow=True)
            self.bf = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,)),dtype=theano.config.floatX),name="bf",borrow=True)
            self.bo = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,)),dtype=theano.config.floatX),name="bo",borrow=True)
            self.bu = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,)),dtype=theano.config.floatX),name="bu",borrow=True)
            self.bs = theano.shared(value=np.asarray(
                rng.uniform(size=(self.num_classes,)),dtype=theano.config.floatX),name="bs",borrow=True)
            self.Wi = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,in_dim)),dtype=theano.config.floatX),name="Wi",borrow=True)
            self.Wf = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,in_dim)),dtype=theano.config.floatX),name="Wf",borrow=True)
            self.Wo = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,in_dim)),dtype=theano.config.floatX),name="Wo",borrow=True)
            self.Wu = theano.shared(value=np.asarray(
                rng.uniform(size=(self.mem_dim,in_dim)),dtype=theano.config.floatX),name="Wu",borrow=True)
            self.Ws = theano.shared(value=np.asarray(rng.uniform(size=(
                self.num_classes,self.mem_dim)),dtype=theano.config.floatX),name="Ws",borrow=True)
            self.Ui_R = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Ui_R",borrow=True)
            self.Ui_L = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Ui_L",borrow=True)
            self.Uf_R = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uf_R",borrow=True)
            self.Uf_L = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uf_L",borrow=True)
            self.Uo_R = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uo_R",borrow=True)
            self.Uo_L = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uo_L",borrow=True)
            self.Uu_R = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uu_R",borrow=True)
            self.Uu_L = theano.shared(value=np.asarray(rng.uniform(size=(
                self.mem_dim,self.mem_dim)),dtype=theano.config.floatX),name="Uu_L",borrow=True)
        else:
            self.load_param(param_file)

        self.params= [self.Wi,self.Wf,self.Wo,self.Wu,self.Ws,
                      self.bi,self.bf,self.bo,self.bu,self.bs,
                      self.Ui_R,self.Uf_R,self.Uo_R,self.Uu_R,
                      self.Ui_L,self.Uf_L,self.Uo_L,self.Uu_L]

        hl = T.dvector('hl')
        hr = T.dvector('hr')
        cr = T.dvector('cr')
        cl = T.dvector('cl')
        x = T.dvector('x')
        pred = T.dmatrix('pred')
        golden = T.dmatrix('golden')
        c = T.dvector('c')
        p = T.dvector('p')

        self.composer_i = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Ui_R,hr) + T.dot(self.Ui_L,hl) + self.bi))
        self.composer_f = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Uf_R,hr) + T.dot(self.Uf_L,hl) + self.bf))
        self.composer_o = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Uo_R,hr) + T.dot(self.Uo_L,hl) + self.bo))
        self.composer_u = theano.function([hr,hl],
                self.outer_activation(T.dot(self.Uu_R,hr) + T.dot(self.Uu_L,hl) + self.bu))
        self.leaf_i = theano.function([x],self.inner_activation(T.dot(self.Wi,x) + self.bi))
        self.leaf_f = theano.function([x],self.inner_activation(T.dot(self.Wf,x) + self.bf))
        self.leaf_o = theano.function([x],self.inner_activation(T.dot(self.Wo,x) + self.bo))
        self.leaf_u = theano.function([x],self.outer_activation(T.dot(self.Wu,x) + self.bu))
        self.get_tanh = theano.function([c],self.outer_activation(c))
        self.combine_c = theano.function([cr,cl],cr + cl) # can define different method
        self.cost_func = theano.function([golden,pred],(pred * T.log(golden)).sum() * -1.0)
        self.softmax = theano.function([p],T.transpose(softmax(T.dot(self.Ws,p) + self.bs)))

    def forward_pass(self,sentence,label):
        """
        Given sentence, forward pass
        """
        inpt_tree = self.mgr.get_tree(sentence)
        golden = label
        one_hot_golden = np.zeros(shape=(self.num_classes,1))
        one_hot_golden[golden] = 1
        stack = self.mgr.get_tree_stack(sentence)
        node_hidden = [np.zeros(shape=self.mem_dim)] * (len(stack) + 1)
        node_c = [np.zeros(shape=self.mem_dim)] * ( len(stack) + 1)
        #level-order traversal
        for node in stack:
            if node.is_leaf():
                # print(node.word)
                x = self.mgr.get_glove_vec(node.word)
                node_c[node.idx] = self.leaf_i(x) * self.leaf_u(x)
                node_hidden[node.idx] = node_c[node.idx]*self.get_tanh(node_c[node.idx])
                # node_hidden[node.idx] = self.leaf_o(x) * self.outer_activation(node_c[node.idx])
                # node_hidden[node.idx] = self.leaf_o(node_c[node.idx])
                # print(node_c[node.idx])
                # print(node_hidden[node.idx])
            else:
                child_l,child_r = node.get_child()
                node_c[node.idx]=((self.composer_i(node_hidden[child_r.idx],node_hidden[child_l.idx])*
                        self.composer_u(node_hidden[child_r.idx],node_hidden[child_l.idx]))+
                        (self.composer_f(node_hidden[child_r.idx],node_hidden[child_l.idx]) *
                        self.combine_c(node_c[child_r.idx],node_c[child_l.idx])))
                node_hidden[node.idx]=(self.composer_o(node_hidden[child_r.idx],
                                node_hidden[child_l.idx])*self.get_tanh(node_c[node.idx]))
                # node_c[node.idx]=((self.composer_i(node_c[child_r.idx],node_c[child_l.idx])*
                        # self.composer_u(node_c[child_r.idx],node_c[child_l.idx]))+
                        # (self.composer_f(node_c[child_r.idx],node_c[child_l.idx]) *
                        # self.combine_c(node_c[child_r.idx],node_c[child_l.idx])))
                # node_hidden[node.idx]=(self.composer_o(node_c[child_r.idx],node_c[child_l.idx]))
        #apply softmax
        pred = self.softmax(node_hidden[inpt_tree.root.idx])
        # pred = self.softmax(node_c[inpt_tree.root.idx])
        self.error = categorical_crossentropy(one_hot_golden,pred)
        # print(self.error)
        return self.error

if __name__ == "__main__":
    pass
