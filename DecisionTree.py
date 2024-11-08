import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
from Node import *

class DecisionTree:
    def __init__(self, max_depth=None, target_column_index=-1):
        self.max_depth = max_depth 
        self.root = None
        self.target_column_index = target_column_index

    def entropy(self, data):
        entropy = 0
        n = len(data)
        classes = data.iloc[:, self.target_column_index]
        classes_count = classes.value_counts()

        for count in classes_count:
            p = count / n
            entropy -= p * np.log2(p)

        return entropy

    def split_data(self, data, column, threshold):
        left = data[data[column] <= threshold]
        right = data[data[column] > threshold]
        return left, right

    def find_best_split(self, data, thresholds):
        best_gain = -1
        best_split = None

        for column in data.columns.drop(data.columns[self.target_column_index]):
            left, right = self.split_data(data, column, thresholds[column])
            gain = self.information_gain(data, left, right)

            if gain > best_gain:
                best_gain = gain
                best_split = (column, thresholds[column], left, right)

        if best_gain == 0:
            return None
        
        return best_split



    def information_gain(self, parent, left, right):
        parent_entropy = self.entropy(parent)
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        weighted_entropy = weight_left * self.entropy(left) + weight_right * self.entropy(right)

        return parent_entropy - weighted_entropy

    def build_tree(self, data, thresholds, depth=0):
        classes = data.iloc[:, self.target_column_index]
        
        if len(classes.unique()) == 1 or (self.max_depth is not None and depth >= self.max_depth): 
            leaf_class = classes.mode()[0]
            return Node(leaf_class=leaf_class)

        split = self.find_best_split(data, thresholds)
        
        if split is None:
            leaf_class = classes.mode()[0]
            return Node(leaf_class=leaf_class)

        split_column, threshold, left, right = split
        left_subtree = self.build_tree(left, thresholds, depth + 1)
        right_subtree = self.build_tree(right, thresholds, depth + 1)
        return Node(split_column=split_column, threshold=threshold, left=left_subtree, right=right_subtree)
    
    def fit(self, data):
        data = data.dropna()

        thresholds = {column: data[column].median() for column in data.columns.drop(data.columns[self.target_column_index])}
        self.root = self.build_tree(data, thresholds)

    def split_data(self, data, column, threshold):
        left = data[data[column] <= threshold]
        right = data[data[column] > threshold]
        return left, right

    def calculate_score(self, test_data):
        correct_predictions = 0

        for _, row in test_data.iterrows():

            actual = row.iloc[self.target_column_index]
            if self.predict_row(row, self.root) == actual:
                correct_predictions += 1

        return correct_predictions / len(test_data)

    

    def predict_row(self, row, node=None):
        if node is None:
            node = self.root

        if node.is_leaf():
            return node.leaf_class
        
        if row[node.split_column] <= node.threshold:
            return self.predict_row(row, node.left)
        
        else:
            return self.predict_row(row, node.right)
    
    
    def plot_tree(self, node, graph=None, pos=None, depth=0, parent=None, label="", pos_offset=1.0, pos_x=0.5):
        if graph is None:
            graph = nx.DiGraph()
            pos = {}

        if node.leaf_class is not None:
            graph.add_node(id(node), label=f"Class: {node.leaf_class}")
        else:
            graph.add_node(id(node), label=f"{node.split_column} <= {node.threshold:.2f}")

        pos[id(node)] = (pos_x, -depth)

        if parent is not None:
            graph.add_edge(parent, id(node), label=label)

        if node.left:
            self.plot_tree(node.left, graph, pos, depth + 1, id(node), "Left", pos_offset / 2, pos_x - pos_offset)
        
        if node.right:
            self.plot_tree(node.right, graph, pos, depth + 1, id(node), "Right", pos_offset / 2, pos_x + pos_offset)

        return graph, pos
    
    def draw_tree(self):
        graph, pos = self.plot_tree(self.root)
        labels = nx.get_node_attributes(graph, 'label')
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=1800, node_color="lightblue", font_size=8, arrows=True)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'label'))
        plt.show()
        

    