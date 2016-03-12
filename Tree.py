#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: Tree.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
class Node(object):
    def __init__(self,word):
        self.word = word
        self.r_child = None
        self.l_child = None

    def __repr__(self):
        return self.word

    def get_child(self):
        return r_child,l_child


class Tree(object):
    # def __init__(self,description):
        # pass
    def __init__(self,root):
        self.root = root
        self.traversal_list = []

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
    root = Node("Buy and Accorsi")
    tree = Tree(root)
    root.r_child = Node("Accorsi")
    root.l_child = Node("Buy and")
    root.l_child.r_child = Node("and")
    root.l_child.l_child = Node("Buy")

    tree.level_order_traversal()
    print(tree.traversal_list)
if __name__ == "__main__":
    main()
