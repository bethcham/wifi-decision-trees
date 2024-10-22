import numpy as np
import matplotlib.pyplot as plt

def entropy(labels):
    l, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # add small value to avoid log(0)

def information_gain(dataset, split_index, split_value):
    left_split = dataset[dataset[:, split_index] <= split_value]
    right_split = dataset[dataset[:, split_index] > split_value]
    
    total_entropy = entropy(dataset[:, -1])
    left_entropy = entropy(left_split[:, -1])
    right_entropy = entropy(right_split[:, -1])
    
    left_weight = len(left_split) / len(dataset)
    right_weight = len(right_split) / len(dataset)
    
    weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
    return total_entropy - weighted_entropy

def find_best_split(dataset):
    best_gain = -1
    best_split_value = None
    best_split_index = None
    best_left_split = None
    best_right_split = None
    
    for col in range(dataset.shape[1] - 1):  # loop over each attribute
        sorted_dataset = dataset[np.argsort(dataset[:, col])]  # sort by attribute
        labels = sorted_dataset[:, -1]  # labels are in last column
        
        for i in range(1, len(dataset)):
            if labels[i] != labels[i - 1]:  # only consider splits that separate different classes
                split_value = (sorted_dataset[i, col] + sorted_dataset[i - 1, col]) / 2
                gain = information_gain(dataset, col, split_value)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split_value = split_value
                    best_split_index = col
                    best_left_split = dataset[dataset[:, col] <= split_value]
                    best_right_split = dataset[dataset[:, col] > split_value]
    
    return best_split_index, best_split_value, best_left_split, best_right_split, round(best_gain, 3)

def decision_tree_learning(training_data, depth):
    labels = training_data[:, -1]
    
    # if all samples have the same label
    if np.all(labels == labels[0]):
        return ({'label' : labels[0]}, depth)
    else:
        split_index, split_value, left_split, right_split, info_gain = find_best_split(training_data)
        
        # recursively build the tree
        l_branch, l_depth = decision_tree_learning(left_split, depth + 1)
        r_branch, r_depth = decision_tree_learning(right_split, depth + 1)
        
        # a new decision tree with root as split value
        node = {
            "split_index": split_index,
            "split_value": split_value,
            "left": l_branch,
            "right": r_branch,
            "gain": info_gain
        }

        return (node, max(l_depth, r_depth))

def plot_tree(tree, depth=0, x=0.5, y=1.0, x_offset=0.7, y_offset=0.15, ax=None):
    if ax is None:
        ax = plt.gca()
    
    # check if it is a leaf node
    if "label" in tree:
        label = f"R{tree['label']}"
        ax.text(x, y, label, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black"))
        return
    
    # plot split condition
    label = f"X{tree['split_index']} <= {tree['split_value']:.2f}"
    ax.text(x, y, label, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", edgecolor="black"))
    
    plot_tree(tree['left'], depth + 1, x - x_offset / (depth + 1), y - y_offset, x_offset, y_offset, ax)
    plot_tree(tree['right'], depth + 1, x + x_offset / (depth + 1), y - y_offset, x_offset, y_offset, ax)
    
    # draw lines 
    ax.plot([x, x - x_offset / (depth + 1)], [y, y - y_offset], 'k-')
    ax.plot([x, x + x_offset / (depth + 1)], [y, y - y_offset], 'k-')

dataset = np.loadtxt("wifi_db/clean_dataset.txt")
tree, depth = decision_tree_learning(dataset, 0)

print("Decision Tree:", tree)
print("Depth: ", depth)

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title("Decision Tree")
plot_tree(tree)
ax.axis("off")
plt.show()