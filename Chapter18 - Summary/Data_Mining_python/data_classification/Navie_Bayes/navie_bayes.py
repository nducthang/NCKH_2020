from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import math as mt
import operator

# source : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
df_data = pd.read_csv("pima-indians-diabetes.csv")


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return mt.sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = mt.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (mt.sqrt(2 * mt.pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


def predict_prob(summaries, X_test):
    y_predict = []
    for i in range(len(X_test)):
        probabilities = calculate_class_probabilities(summarize, X_test[i])
        y_predict.append(max(probabilities.items(), key=operator.itemgetter(1))[0])
    return y_predict


def acc_model(y_test, y_predict):
    count = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            count = count + 1
    return count / len(y_test)


if __name__ == '__main__':
    train, test = train_test_split(df_data, test_size=0.2)
    data_train = train.values
    # print(data_train)
    attribute_names = list(df_data.columns)
    attribute_names.remove("Class")
    X_test = test[attribute_names].values
    # print(X_test.shape)
    y_test = test["Class"].to_list()
    # print(y_test)
    # print(len(y_test))
    summarize = summarize_by_class(data_train)
    y_predict = predict_prob(summarize, X_test)
    print(y_predict)
    print("Accuracy is model:", acc_model(y_test, y_predict)*100, "%")
