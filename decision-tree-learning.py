import numpy as np
import matplotlib

def decision_tree_learning(training_dataset, depth):
    if : # if all samples have the same label
        return # (a leaf node with this value, depth)
    else:
        # split <- find_split(training_dataset)
        # node <- a new decision tree with root as split value
        #l_branch, l_depth <- decision_tree_learning(l_dataset, depth+1)
        #r_branch, r_depth <- decision_tree_learning(r_dataset, depth+1)
        return (node, max(l_depth, r_depth))
    
def find_split(dataset):
    # find split that gives the highest information gain

data = np.loadtxt("wifi_db/clean_dataset.txt")
decision_tree_learning(data, )