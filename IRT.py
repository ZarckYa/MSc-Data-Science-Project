#!/usr/bin/env python
# coding: utf-8


import re
import numpy as np
import random
from tqdm import tqdm

from catsim.initialization import Initializer
from catsim.selection import MaxInfoSelector
from catsim.selection import UrrySelector
from catsim.irt import normalize_item_bank
from catsim.initialization import RandomInitializer
from catsim.simulation import Simulator
from catsim.stopping import MaxItemStopper
from catsim.estimation import NumericalSearchEstimator
from catsim.cat import generate_item_bank
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 

from girth import ability_eap, ability_map, ability_mle
from girth import multidimensional_ability_eap, multidimensional_ability_map

from readability import Readability
from sklearn.preprocessing import minmax_scale


def getMaterialDifficulty(dataf):
    
    materialsDifficultyLevel = []
    # questionsDifficultyLevel = []

    for data in tqdm(dataf):
        conv = list(data.keys())[0]
        conv = re.sub(r'\(*\d*\)', r'',conv)
        
        ques = list(data.values())[0]
        lenQues = len(ques)
    #     ques = list(data.values())[0]

        if len(conv.split()) < 100:
            materialsDifficultyLevel.append(1 + (0.5*lenQues))
        else:
            readability = Readability(conv)
            readabilityLevel = readability.dale_chall()
            materialsDifficultyLevel.append(readabilityLevel.score + (0.5*lenQues))   
    
    # materialDifficultyLevelNor = np.divide(materialsDifficultyLevel, max(materialsDifficultyLevel))
    materialDifficultyLevelNor = minmax_scale(materialsDifficultyLevel, feature_range=(-2, 2))

    return materialDifficultyLevelNor

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




class Basic_IRT_Adptive_Recommendation():
    
    def __init__(self, numberOfMaterials, materialDifficultyLevelNor, questionsDifficultyLevelNor):
        self.administered_items  = []
        self.selected_items_once = []
        self.selected_all_items = []
        self.response_all = []
        if len(np.array(materialDifficultyLevelNor).shape) == 1:
            self.item_parameter = normalize_item_bank(np.array([materialDifficultyLevelNor]).T)
        else:
            self.item_parameter = normalize_item_bank(np.array(materialDifficultyLevelNor))
        self.ability = 0
        self.whole_ability = 0
        self.numberOfMaterials = numberOfMaterials
        self.questionsDifficultyLevelNor = questionsDifficultyLevelNor

    def normalize_item_parameter(self, one_item_array):
        normalize_item = normalize_item_bank(one_item_array.T)
        return normalize_item
        
    def initial_ability(self):
        # init_ability = RandomInitializer(dist_params=(-2,2)).initialize()
        init_ability = random.uniform(-2,2)
        self.ability = init_ability
        self.whole_ability = init_ability
        return init_ability
    
    def IRT_ability_estimator(self, response_vector):
        item_once = []
        item_all = []
        for i in self.selected_items_once:
            item_once.extend(self.questionsDifficultyLevelNor[i])

        for i in self.selected_all_items:
            item_all.extend(self.questionsDifficultyLevelNor[i])        

        if len(np.array(self.questionsDifficultyLevelNor[0]).shape) == 2:
            item_once = np.array(item_once)
            item_all = np.array(item_all)
            items_parameter_once = normalize_item_bank(item_once)
            items_parameter_all = normalize_item_bank(item_all)
        else:
            item_once = np.array([item_once])
            item_all = np.array([item_all])
            items_parameter_once = normalize_item_bank(item_once.T)
            items_parameter_all = normalize_item_bank(item_all.T)

        
        difficulty_once = items_parameter_once[:,1]
        difficulty_all = items_parameter_all[:,1]

        discrimination_once = items_parameter_once[:,0]
        discrimination_all = items_parameter_all[:,0]

        # print("item parameter: ", items_parameter_all)
        # print("diff: ", difficulty_all)
        # print("dis: ", discrimination_all)

        response_once = response_vector

        self.response_all.extend(response_once)

        # selected_items = np.array([i for i in range(len(item))])

        # Estimetor = NumericalSearchEstimator()
        # newAbility = Estimetor.estimate(items=items_parameter, administered_items=selected_items, 
        #                                 response_vector=respones, est_theta=self.ability)   

        new_ability_once = ability_map(dataset = response_once, difficulty = difficulty_once, discrimination = discrimination_once)
        new_ability_all = ability_map(dataset = np.array(self.response_all), difficulty = difficulty_all, discrimination = discrimination_all)

        # self.ability = newAbility
        self.ability = new_ability_once[0]
        self.whole_ability = new_ability_all[0]

        return self.ability, self.whole_ability

    def MIRT_ability_estimator(self, response_vector):
        item_once = []
        item_all = []
        for i in self.selected_items_once:
            item_once.extend(self.questionsDifficultyLevelNor[i])

        for i in self.selected_all_items:
            item_all.extend(self.questionsDifficultyLevelNor[i])        

        item_once = np.array([item_once])
        item_all = np.array([item_all])

        items_parameter_once = normalize_item_bank(item_once.T)
        items_parameter_all = normalize_item_bank(item_all.T)

        difficulty_once = items_parameter_once[:,1]
        difficulty_all = items_parameter_all[:,1]

        discrimination_once = items_parameter_once[:,0]
        discrimination_all = items_parameter_all[:,0]
        

        response_once = response_vector

        self.response_all.extend(response_once)

        # selected_items = np.array([i for i in range(len(item))])

        # Estimetor = NumericalSearchEstimator()
        # newAbility = Estimetor.estimate(items=items_parameter, administered_items=selected_items, 
        #                                 response_vector=respones, est_theta=self.ability)   

        new_ability_once = multidimensional_ability_eap(dataset = response_once, difficulty = difficulty_once, discrimination = discrimination_once)
        new_ability_all = multidimensional_ability_eap(dataset = np.array(self.response_all), difficulty = difficulty_all, discrimination = discrimination_all)

        # self.ability = newAbility
        self.ability = new_ability_once[0]
        self.whole_ability = new_ability_all[0]

        return self.ability, self.whole_ability


    def item_selector(self):
        self.selected_items_once = []
        for _ in range(self.numberOfMaterials):
            selector = UrrySelector()
            new_selected_item = selector.select(items=self.item_parameter, administered_items=self.administered_items, 
                                                est_theta=self.whole_ability)
            self.administered_items.append(new_selected_item)
            self.selected_items_once.append(new_selected_item)
            self.selected_all_items.append(new_selected_item)
        return self.selected_items_once

