# Code written By Eshwar

from sklearn.datasets import load_breast_cancer
import numpy as np
import random

def calculate_entropy(p, n):
  total = p+n
  return (((-p/total)*(np.log2(p/total)))+((-n/total)*(np.log2(n/total))))

def calculate_threshold_entropy(classLabel):
  p = n = 0
  for i_cl in classLabel:
    if i_cl==0:
      n+=1
    else:
      p+=1
  return (p,n)

def calculate_node(newX, features_col_num):
  IG_indv_feature = []
  best_threshold = []
  systemEntropy = calculate_entropy(212, 357) #calculating total system's entropy
  for i in features_col_num:
    sorted_X = np.array(sorted(newX, key=lambda x: x[i]))
    indv_feature_data = sorted_X[:,i]

    left = right = []
    index_break = 0
    IG_best = 0
    bt = 0
    best_index_break = 0

    for j in range(len(indv_feature_data)-1):
      threshold = indv_feature_data[j]+(indv_feature_data[j+1]-indv_feature_data[j])/2
      for k in range(len(indv_feature_data)):
        if threshold<indv_feature_data[k]:
          index_break=k-1
          break

      left = newX[:index_break,-1]
      right = newX[index_break:,-1]
      leftThresholdEntropy = rightThresholdEntropy = 0
      
      #-------------- Left Entropy----------
      pl, nl = calculate_threshold_entropy(left)
      if(pl>0 and nl>0):
        leftThresholdEntropy = calculate_entropy(pl, nl)
      
      #-------------- Right Entropy----------
      pr, nr = calculate_threshold_entropy(right)
      if(pr>0 and nr>0):
        rightThresholdEntropy = calculate_entropy(pr, nr)

      #Information gain of single feature on each threshold  
      IG_temp = systemEntropy - ((((pl+nl)/569)*leftThresholdEntropy) + (((pr+nr)/569)*rightThresholdEntropy))
      if(IG_temp>IG_best):
        IG_best = IG_temp
        bt= threshold
        best_index_break = index_break

    best_threshold.append(bt)
    IG_indv_feature.append([IG_best,i,bt,best_index_break])
  
  indv_node_data = (sorted(IG_indv_feature, key=lambda x:x[0], reverse=True))
  indv_node_data = indv_node_data[0][1:]
  TreeData.append(indv_node_data)

  sorted_X_on_selected_feature = np.array(sorted(newX, key=lambda x: x[indv_node_data[0]]))
  selected_left = sorted_X_on_selected_feature[:indv_node_data[2]]
  selected_right = sorted_X_on_selected_feature[indv_node_data[2]:]
  return selected_left, selected_right, indv_node_data[0]

def main_call(queue, features_col_num):
  while(features_col_num):
    queue_data = queue.pop(0)
    returnedLeft, returnedRight, remove_feature_col_num = calculate_node(queue_data, features_col_num)
    if(len(returnedLeft)>0):
      cl = returnedLeft[:,-1]
      one = zero = 0
      for cl_val in cl:
        if cl_val==0:
          zero += 1
        else:
          one += 1
      if(zero==0 or one==0):
        if(zero==0):
          TreeData.append(1)
        else:
          TreeData.append(0)
      if(zero>0 and one>0):
        queue.append(returnedLeft)

    if(len(returnedRight)>0):
      cl = returnedRight[:,-1]
      one = zero = 0
      for cl_val in cl:
        if cl_val==0:
          zero += 1
        else:
          one += 1
      if(zero==0 or one==0):
        if(zero==0):
          TreeData.append(1)
        else:
          TreeData.append(0)
      if(zero>0 and one>0):
        queue.append(returnedRight)
      
    features_col_num.remove(remove_feature_col_num)
# Node Decleration
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def final_tree(arr, root, s, e):
    if s < e:
        temp = Node(arr[s])
        root = temp
        root.left = final_tree(arr, root.left, 2 * s + 1, e)
        root.right = final_tree(arr, root.right, 2 * s + 2, e)
    return root

def preOrder(root):
    if root != None:
        print(root.data,end=" ")
        preOrder(root.left)
        preOrder(root.right)

# Testing Data by Traversing Tree
def tree_traversal(root, ans, testData1):
  flag=0
  if(root):
    if(isinstance(root.data, int)):
      if(root.data==testData1[-1]):
        flag=1
        return 1
      else:
        flag=1
        return 0

    else:
      if(root.left is None and root.right is None):
        if(flag==0):
          return 1
      if(testData1[root.data[0]]<root.data[1]):
        return(tree_traversal(root.left, ans, testData1))
      else:
        return(tree_traversal(root.right, ans, testData1))


data = load_breast_cancer()

X=data['data']
y=data['target']

newX=np.append(X, y[:,None], axis=1)

newX = newX[:int(len(newX)*0.7)]
TreeData = []
features_col_num = [i for i in range(30)]
queue = []

queue.append(newX)

main_call(queue, features_col_num)

# print(TreeData)

final_arr = []
for i in TreeData:
  if(isinstance(i, int)):
    final_arr.append(i)
  else:
    final_arr.append([i[0],i[1]])

# Building Tree
n = len(final_arr)
root = None
root = final_tree(final_arr, root, 0, n)
temp_root=root
print("Tree structure: ",end=" ")
preOrder(root)
print("")
print("")
# Testing
def valid_test(type_of_data, test_data):
  test_final_arr = []  
  for test_indv_data in test_data:
    test_final_arr.append(tree_traversal(temp_root, 0, test_indv_data))

  # Calculating accuracy
  test_sum = sum(test_final_arr)
  test_total = len(test_final_arr)

  accuracy = (test_sum*100)/test_total
  
  print("Accuracy of",type_of_data, "data: ", accuracy)


newX=np.append(X, y[:,None], axis=1)
valid_data = newX[int(len(newX)*0.7):int(len(newX)*0.8)] # validation data 10%
test_data = newX[int(len(newX)*0.8):] # testing data 20%
valid_test("Validation", valid_data)
valid_test("Testing", test_data)

################################################################################
# Calculating the height of the tree

def calculate_height_of_tree(root):
  if root is None:
    return 0
  else:
    left_height = calculate_height_of_tree(root.left)
    right_height = calculate_height_of_tree(root.right)

    if(left_height > right_height):
      return left_height + 1
    else:
      return right_height + 1


tree_height_to_be_pruned = calculate_height_of_tree(temp_root)


# Pre-Pruning the tree by limiting the height of tree
def prune_tree(root, h, n):
  if root is None:
        return 0
  else:
      ldepth = prune_tree(root.left, h, n)
      rdepth = prune_tree(root.right, h, n)
      
      if(n-max(ldepth, rdepth)>=h):
          root.left = None
          root.right = None
          
      if(ldepth > rdepth):
          return ldepth + 1
      else:
          return rdepth + 1

  if root is None:
    return 0
  else:
    left_height = calculate_height_of_tree(root.left)
    right_height = calculate_height_of_tree(root.right)

    if(left_height>height or right_height>height):
      root = None
    if(left_height > right_height):
      return left_height + 1
    else:
      return right_height + 1

prune_tree(temp_root, tree_height_to_be_pruned-2, tree_height_to_be_pruned)
prune_root = root

# Passing the same validation data and test data
# as we passed in the above generated decision tree to calculate the accuracies.
#The accuracies did improve by a factor of almost 10%.
def prune_tree_traversal(root, ans, testData1):
  flag=0
  if(root):
    if(isinstance(root.data, int)):
      if(root.data==testData1[-1]):
        flag=1
        return 1
      else:
        flag=1
        return 0

    else:
      if(root.left is None and root.right is None):
        if(flag==0):
          return 1
      if(testData1[root.data[0]]<root.data[1]):
        return(prune_tree_traversal(root.left, ans, testData1))
      else:
        return(prune_tree_traversal(root.right, ans, testData1))


def prune_valid_test(type_of_data, test_data):
  test_final_arr = []  
  for test_indv_data in test_data:
    test_final_arr.append(tree_traversal(prune_root, 0, test_indv_data))

  # Calculating accuracy
  test_sum = sum(test_final_arr)
  test_total = len(test_final_arr)

  accuracy = (test_sum*100)/test_total
  
  print("Accuracy of",type_of_data, "data after Pruning: ", accuracy)


newX=np.append(X, y[:,None], axis=1)
valid_data = newX[int(len(newX)*0.7):int(len(newX)*0.8)] # validation data 10%
test_data = newX[int(len(newX)*0.8):] # testing data 20%
prune_valid_test("Validation", valid_data)
prune_valid_test("Testing", test_data)
