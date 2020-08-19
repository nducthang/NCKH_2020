import pandas as pd
import math as mt
from collections import Counter

df_tennis = pd.read_csv("weather.csv")
print(df_tennis)


# Function to calculate the entropy of probaility of observations
def entropy(probs):
    return sum([-prob * mt.log(prob, 2) for prob in probs])


# Function to calulate the entropy of the given Data Sets/List with respect to target attributes
def entropy_of_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)
    probs = [x/num_instances for x in cnt.values()]
    return entropy(probs)


# Information gain of Attributes
def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    # split data by possible vals of attribute
    df_split = df.groupby(split_attribute_name)
    # proportion of Obs in Each data_split
    nobs = len(df.index)
    df_agg_ent = df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x) / nobs]})[
        target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    # Calculate Information Gain:
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


# ID3 Algorithm
def id3_algorithm(df, target_attribute_name, attribute_names, default_class=None):
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class
    else:
        # Get Default Value for next recursive call of this function:
        default_class = max(cnt.keys())
        # Compute the information gain of the attribute:
        gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
        # index of best attribute
        index_of_max = gainz.index(max(gainz))
        # choose best attribute to split on
        best_attr = attribute_names[index_of_max]
        # create an empty tree, to be populated in a moment
        # Iniiate the tree with best attribute as a node
        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3_algorithm(data_subset, target_attribute_name, remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree
        return tree


# Classification accuracy
def classify(instance, tree, default=None):
    # instance of play tennis with predict
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():  # Value of the attributs in  set of Tree keys
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):  # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result  # this is a label
    else:
        return default


if __name__ == '__main__':
    print('Information_gain for Outlook is :', (information_gain(df_tennis, 'outlook', 'play')))
    attribute_names = list(df_tennis.columns)
    print("List of Attributes:", attribute_names)
    # Remove the class attribute
    attribute_names.remove('play')
    attribute_names.remove('id')
    print("Predicting Attributes:", attribute_names)
    tree = id3_algorithm(df_tennis, 'play', attribute_names)
    print(tree)
    print("Decision tree predict:")
    df_tennis['predicted'] = df_tennis.apply(classify, axis=1, args=(tree, 'No'))
    print(df_tennis['predicted'])