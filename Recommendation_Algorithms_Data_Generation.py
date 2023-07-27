#!/usr/bin/env python
# coding: utf-8



import numpy as np
import json
import pandas as pd
import re
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm



from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer



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
def Logistic_Regression(xtrain, ytrain, xtest):
    ytrain_0_1 = []
    lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
    for y in ytrain:
        if y > 0.65:
            ytrain_0_1.append(1)
        else:
            ytrain_0_1.append(0)
    lr_model = lr.fit(xtrain, ytrain_0_1)
    y_predict_prob = lr_model.predict_proba(xtest)
    y_predict_label = lr_model.predict(xtest)
    return lr_model, y_predict_prob

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


def Max_N_Expectation(scores, questions_probs, posterior_probs, n_expectatios):
    expectation_x_z = []
    if len(posterior_probs[1]) > 1:
        posterior_y1_probs = posterior_probs[:,1]
    else:
        posterior_y1_probs = posterior_probs[:,0]
    for i, _ in enumerate(scores):
        E = scores[i] * questions_probs[i] * posterior_y1_probs[i]
        expectation_x_z.append(E)
    
    expectation_x_z = np.argsort(expectation_x_z)
#     print(expectation_x_z)
    top_n_ques_index = expectation_x_z[(len(expectation_x_z) - n_expectatios):]
    return top_n_ques_index


