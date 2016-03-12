#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: Tree.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
from __future__ import print_function
import sys

class Node(object):
    def __init__(self,word="",idx=None):
        self.word = word
        self.r_child = None
        self.l_child = None
        self.idx= idx

    def __repr__(self):
        return self.word

    def get_num_child(self):
        if self.l_child is None : assert(self.r_child is None)
        return 2-[self.r_child,self.l_child].count(None)

    def is_leaf(self):
        return (self.get_num_child() == 0)

    def get_child(self):
        return self.l_child,self.r_child


class Tree(object):
    def __init__(self,word_seq,description):
        """
        sentence : [The food is good .]
        description will like [6 6 8 8 9 7 0 9 7]
        """
        #node_list[0] is dummy
        self.traversal_list=[]
        self.node_list = [Node() for i in range(len(description) + 1)]#shouldn't use []*10(s copy)
        for i in range(1,len(self.node_list)):
            self.node_list[i].idx = i

        for i,word in enumerate(word_seq,1):
            n = Node(word,i)
            self.node_list[i] = n
        for i,des in enumerate(description,1):
            if des == 0:
                self.root = self.node_list[i]
            else:
                n_child = self.node_list[des].get_num_child()
                if n_child == 1: #left child has been assigned!
                    self.node_list[des].r_child = self.node_list[i]
                    self.node_list[des].word += self.node_list[i].word
                elif n_child == 0:
                    self.node_list[des].l_child = self.node_list[i]
                    self.node_list[des].word += (self.node_list[i].word + " ")
                else:
                    print(" i:{} des:{} went wrong".format(i,des),file=sys.stderr)
        #Do level-order traversal
        self.level_order_traversal()


    def level_order_traversal(self):
        if self.root is None:
            return list()

        queue=[]
        queue.append(self.root)

        while(len(queue) > 0):
            node = queue.pop(0)
            self.traversal_list.append(node)

            if node.l_child is not None:
                queue.append(node.l_child)
            if node.r_child is not None:
                queue.append(node.r_child)

        self.traversal_list.reverse()

def main():
    # word_seq = ["The","noodles","taste","great","*","but","the","service","is","bad","."]
    # description=[12,12,18,18,14,15,19,19,21,21,17,13,14,15,16,17,0,13,20,16,20]
    # tree1 = Tree(word_seq,description)
    # print(tree1.traversal_list)
    # for i in range(1,len(tree1.node_list)):
        # print(tree1.node_list[i])
    word_seq = ["The","food","is","good","."]
    description=[6,6,8,8,9,7,0,9,7]
    tree2 = Tree(word_seq,description)
    print(tree2.node_list)
if __name__ == "__main__":
    main()
