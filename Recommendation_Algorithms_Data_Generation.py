#!/usr/bin/env python
# coding: utf-8


import copy
import numpy as np
import json
import pandas as pd
import re
import tensorflow as tf

from numpy.linalg import norm
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from GUI_Design import data_to_GUI

# from gensim.models import LdaModel
# from gensim.parsing.preprocessing import STOPWORDS
# from gensim.corpora import Dictionary
# from gensim.utils import simple_preprocess
# from nltk.stem import WordNetLemmatizer



def loadMaterialsFeaturesForML(path):
    MaterialsFeaturesForML = pd.read_csv(path, index_col=False)
    return MaterialsFeaturesForML


def getQuestionsDifficulty(dataf, materialDifficultyLevelNor):
    questionsDifficultyLevelNor = []
    if len(np.array(materialDifficultyLevelNor).shape) == 1:
        for i, data in tqdm(enumerate(dataf)):
            ques = list(data.values())[0]   
            questionsDifficultyLevelNor.append([materialDifficultyLevelNor[i]]*len(ques))  
    else:
        for i, data in tqdm(enumerate(dataf)):
            ques = list(data.values())[0]   
            questionsDifficultyLevelNor.append([list(materialDifficultyLevelNor[i])]*len(ques))
    
    return questionsDifficultyLevelNor

def getQuestionsScore(questionsDifficultyLevelNor):
    question_score = []
    material_score = []
    question_score_int = []
    material_score_int = []    
    for quesdif in questionsDifficultyLevelNor:
        temp = []
        for onequesdif in quesdif:
            temp.append(onequesdif)
        material_score.append(sum(temp))
        question_score.append(temp)
    
    min_scores = (min(material_score) * (-1))+1
    for i, scores in  enumerate(question_score):
        material_score_int.append(material_score[i] + min_scores)
        temp_int = []
        for onescore in scores:
            temp_int.append(onescore + min_scores)
        question_score_int.append(temp_int)
        
    return question_score_int, material_score_int

def getProbOfQuestions(ability, CandidateMaterialsDifficultyLevel):
    basic_prob = 1/len(CandidateMaterialsDifficultyLevel)
    question_center_1_range = int(len(CandidateMaterialsDifficultyLevel)/40)
    question_center_2_range = int(len(CandidateMaterialsDifficultyLevel)/20)
    question_center_3_range = int(len(CandidateMaterialsDifficultyLevel)/10)
    

    diff_ability_close_level = []
    for quesdif in CandidateMaterialsDifficultyLevel:
        diff_ability_close_level.append(abs(quesdif-ability))
    
    diff_ability_close_level_array = np.array(diff_ability_close_level)
    diff_ability_close_level_array_sorted = np.argsort(diff_ability_close_level_array)
    
    _1_probs = 4*(1/len(CandidateMaterialsDifficultyLevel))
    _2_probs = 3*(1/len(CandidateMaterialsDifficultyLevel))
    _3_probs = 2*(1/len(CandidateMaterialsDifficultyLevel))
    
    priority_probs = question_center_1_range*_1_probs + question_center_2_range*_2_probs + question_center_3_range*_3_probs
    edge_probs = (1-priority_probs)/(len(CandidateMaterialsDifficultyLevel) - question_center_1_range - question_center_2_range - question_center_3_range)
        
    probs_distribution = np.zeros((len(CandidateMaterialsDifficultyLevel)))
    
    for i in diff_ability_close_level_array_sorted[:question_center_1_range]:
        probs_distribution[i] = _1_probs 
    
    for i in diff_ability_close_level_array_sorted[question_center_1_range:question_center_2_range]:
        probs_distribution[i] = _2_probs
        
    for i in diff_ability_close_level_array_sorted[question_center_2_range:question_center_3_range]:
        probs_distribution[i] = _3_probs
    
    for i in diff_ability_close_level_array_sorted[question_center_3_range:]:
        probs_distribution[i] = edge_probs
    
    return probs_distribution

# 将已知的数据进行的训练，然后将所有的推荐候选的题目(特征)输入进逻辑回归模型，模型将会进行二分类，最终模型将会返回每个题目的y=1/0的概率，
# 获取所有的题目的y=1的概率，然后结合分数（Si）以及题目概率(P(i))，计算期望。

# 题目特征，初步定义为：1. 一个听力材料的试题数量。 2. 听力材料的主题类别。 3.听力试题的难度。

# Train on the known data, then input all the recommended candidate questions (features) into a logistic regression model, 
# which will perform binary classification, and ultimately the model will return a probability of y=1/0 for each question.
# Get the probability of y=1 for all topics, then combine the score (Si) and the topic probability (P(i)) to calculate the expectation.

# The topic characteristics, initially defined as 1. the number of test questions in a piece of listening material. 
# 2. the topic category of the listening material. 3. the difficulty of the listening test questions.
class Custom_Logistic_Regression():
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        # x = self._transform_x(x)
        # y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (np.ones(len(y_true))-y_true) * np.log(np.ones(len(y_true)) - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def predict_probs(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return probabilities
    
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)


def Logistic_Regression(xtrain, ytrain, xtest):

    ytrain_0_1 = []
    xtrain_0_1 = []
    i_0_1 = []
    # lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    lr = Custom_Logistic_Regression()
    
    for i,y in enumerate(ytrain):
        if y < 0.74:
            ytrain_0_1.append(0)  
        else:
            i_0_1.append(i)
#             ytrain_0_1.append(1)
    
    xtrain_0_1 = np.delete(np.array(xtrain), i_0_1, axis=0)

    # lr_model = lr.fit(xtrain, ytrain_0_1)
    lr.fit(xtrain_0_1, ytrain_0_1, epochs=30)

    y_predict_prob = lr.predict_probs(xtest)
    y_predict_label = lr.predict(xtest)
    return list(y_predict_prob)


#def Logistic_Regression(xtrain, ytrain, xtest):

#    ytrain_0_1 = []
#    xtrain_0_1 = []
#    i_0_1 = []
#    # lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
#    lr = Custom_Logistic_Regression()
    
#    for i,y in enumerate(ytrain):
#        if y < 0.74:
#            ytrain_0_1.append(0)  
#        else:
#            i_0_1.append(i)
#        #    ytrain_0_1.append(1)
    
#    xtrain_0_1 = np.delete(np.array(xtrain), i_0_1, axis=0)

#    # lr_model = lr.fit(xtrain, ytrain_0_1)
#    lr_model = lr.fit(xtrain_0_1, ytrain_0_1, epochs=30)

#    y_predict_prob = lr_model.predict_proba(xtest)
#    y_predict_label = lr_model.predict(xtest)
#    return lr_model, y_predict_prob

def DecisionTreeRegression(xtrain, ytrain, xtest, depth):
    ytrain_0_1 = []
    xtrain_0_1 = []
    i_0_1 = []
    dt = DecisionTreeClassifier(max_depth=depth)
    
    for i,y in enumerate(ytrain):
        if y < 0.74:
            ytrain_0_1.append(0)  
        else:
            i_0_1.append(i)
        #    ytrain_0_1.append(1)
    
    xtrain_0_1 = np.delete(np.array(xtrain), i_0_1, axis=0)
    dt_model = dt.fit(xtrain_0_1, ytrain_0_1)
    y_predict_prob = dt_model.predict_proba(xtest)
    y_predict_label = dt_model.predict(xtest)
    return dt_model, y_predict_prob


def NN(xtrain, ytrain, xtest, lr, decay):
    
    xtrain = np.array(xtrain.values)
    xtest = xtest.values
    ytrain = np.array(ytrain)
    ytrain_0_1 = []
    i_0_1 = []
    for i,y in enumerate(ytrain):
        if y < 0.74:
            ytrain_0_1.append(0)
        else:
            i_0_1.append(i)
#             ytrain_0_1.append(1)
    xtrain_0_1 = np.delete(xtrain, i_0_1, axis=0)
    input_shape = [xtrain.shape[1]]
    model = Sequential([
        Dense(units=32, activation='relu', input_shape=input_shape),
        Dense(units=16, activation='relu'),
        Dense(units=2, activation='softmax')
    ])
    
    adam = Adam(learning_rate = lr, decay = decay)

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')
    
    model.fit(xtrain_0_1, np.array(ytrain_0_1))
    
    result_probs = model.predict(xtest)
    return result_probs

def recommendation_by_right_answers(answers, answers_index, ability, material_features_path):
    answers_1 = []
    xtrain_data_1 = []
    i_0 = []
    i_1 = []
    recommend_index = []
    
    for i,y in enumerate(answers):
        if y > 0.74:
            answers_1.append(1)  
            i_1.append(i)
        else:
            i_0.append(i)
    xtrain_data_1 = np.delete(answers_index, i_0, axis=0)
    
    marterials_features = pd.read_csv(material_features_path, index_col=False)
    correct_items = marterials_features.iloc[i_1]
    display(correct_items)
    for one_data in correct_items[["Material Index", "Materials Difficulty", "Questions Numbers", "Topics"]].values:
        temp_materials = marterials_features.copy()
        temp_distance = []
        for ind, row in enumerate(marterials_features[["Material Index", "Materials Difficulty", "Questions Numbers", "Topics"]].values):
            if (one_data[1] >= row[1]) or (one_data[3] == row[3]) or (row[1] > (ability+0.5)):
                temp_materials = temp_materials.drop(ind)
        print(len(temp_materials))
        for rows in temp_materials[["Materials Difficulty", "Questions Numbers", "Topics"]].values:
            temp_distance.append(np.dot(one_data[1:], rows)/(norm(one_data[1:])*norm(rows)))
#         print(len(temp_materials))
        max_2_index = np.argsort(np.array(temp_distance))[:2]
        max_2_index_df = temp_materials.iloc[list(max_2_index)]
        recommend_index.extend(max_2_index_df["Material Index"])
    return(list(set(recommend_index)))



def Max_N_Expectation(scores, questions_probs, posterior_probs, n_expectatios):
    expectation_x_z = []
    if len(np.array(posterior_probs).shape) > 1:
        if np.array(posterior_probs).shape[1] <= 1:
            posterior_y1_probs = posterior_probs[:,0]
        else:
            posterior_y1_probs = posterior_probs[:,1]
    else:
        posterior_y1_probs = posterior_probs
    for i, _ in enumerate(scores):
        E = scores[i] * questions_probs[i] * posterior_y1_probs[i]
        expectation_x_z.append(E)
    
    expectation_x_z = np.argsort(expectation_x_z)
#     print(expectation_x_z)
    top_n_ques_index = expectation_x_z[(len(expectation_x_z) - n_expectatios):]
    return top_n_ques_index


# To get the recommendation questions
def getRecommendationQuestions(material_feature_path, matDiffPath, dataf, correct_rate, all_selected_items, ability, n_expectations, recom_algo = "Logistic_Regression"):

    matDiff = pd.read_csv(matDiffPath)
    
    quesDiff = getQuestionsDifficulty(dataf, list(matDiff['Materials Difficulty']))
    for i in all_selected_items:
        quesDiff.pop(i)
    
    matDiff = matDiff.drop(index = all_selected_items)
    print(len(quesDiff))
    
    # To get the features of each materials, and split data into train and test data, as input of NN, Logistic or Decision tree.
    material_features = loadMaterialsFeaturesForML(material_feature_path)
    Xtrain = material_features.iloc[all_selected_items]
    Xtrain=Xtrain[["Materials Difficulty", "Questions Numbers", "Topics"]]
    Xtest = material_features.drop(index = all_selected_items)
    Xtest=Xtest[["Materials Difficulty", "Questions Numbers", "Topics"]]
    print(Xtest.shape)
    question_score, material_score = getQuestionsScore(quesDiff)
    print(len(material_score))
    probs_distribution = getProbOfQuestions(ability=ability, CandidateMaterialsDifficultyLevel=list(matDiff['Materials Difficulty']))

    if recom_algo == "Logistic_Regression":
        predict_probs = Logistic_Regression(xtrain=Xtrain, ytrain=correct_rate, xtest=Xtest)
    elif recom_algo == "Decision_tree":
        dt_model, predict_probs = DecisionTreeRegression(xtrain=Xtrain, ytrain=correct_rate, xtest=Xtest, depth=4)
    elif recom_algo == "NN":
        predict_probs=NN(xtrain=Xtrain, ytrain=correct_rate, xtest=Xtest, lr=1e-6, decay=1e-2)
    
    top_n_questions_by_wrong = Max_N_Expectation(scores=material_score, questions_probs=probs_distribution, 
                                    posterior_probs=predict_probs, n_expectatios=n_expectations)
    top_n_questions_by_right = recommendation_by_right_answers(answers=correct_rate, answers_index=all_selected_items, 
                                                               ability=ability, material_features_path=material_feature_path)
    top_n_questions = []
    top_n_questions.extend(top_n_questions_by_wrong)
    top_n_questions.extend(top_n_questions_by_right)
    top_n_questions = list(set(top_n_questions))
    
    Questions, Choices, TrueAnswers, OutputPath = data_to_GUI(Data = dataf, indices = top_n_questions)
    return Questions, Choices, TrueAnswers, OutputPath, top_n_questions, top_n_questions_by_right, top_n_questions_by_wrong