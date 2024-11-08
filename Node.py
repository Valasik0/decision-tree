class Node:
    def __init__(self, split_column=None, threshold=None, left=None, right=None, leaf_class=None):
        self.split_column = split_column  
        self.threshold = threshold        
        self.left = left                  
        self.right = right                
        self.leaf_class = leaf_class
    
    def is_leaf(self):
        return self.leaf_class is not None