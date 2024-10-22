import numpy as np
import matplotlib

def find_split(dataset):
    # find split that gives the highest information gain
    split_value = np.median(dataset[:, 0])
    left_split = dataset[dataset[:, 0] <= split_value]
    right_split = dataset[dataset[:, 0] > split_value]
    return split_value, left_split, right_split

def decision_tree_learning(training_data, depth):
    labels = training_data[:, -1]
    
    if np.all(labels == labels[0]):  # all samples have the same label
        return (labels[0], depth)  # return leaf node
    else:    
        split_value, left_split, right_split = find_split(training_data)
        
        if len(left_split) == 0 or len(right_split) == 0:
            return (labels[0], depth)
        
        # Recursively build the tree
        l_branch, l_depth = decision_tree_learning(left_split, depth + 1)
        r_branch, r_depth = decision_tree_learning(right_split, depth + 1)
        
        return (split_value, max(l_depth, r_depth))
    
clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt")
# print(clean_dataset)

tree, depth = decision_tree_learning(clean_dataset, 0)

print("Decision Tree:", tree)
print("Depth of the tree:", depth)