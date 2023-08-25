#!/usr/bin/env python
# coding: utf-8


import re
import numpy as np
import random
from tqdm import tqdm

from catsim.selection import MaxInfoSelector
from catsim.selection import UrrySelector
from catsim.irt import normalize_item_bank
from catsim.initialization import RandomInitializer
from catsim.simulation import Simulator
from catsim.stopping import MaxItemStopper
from catsim.estimation import NumericalSearchEstimator
from catsim.cat import generate_item_bank

from girth import ability_eap, ability_map, ability_mle
from girth import multidimensional_ability_eap, multidimensional_ability_map
from girth_mcmc import GirthMCMC

from scipy.special import roots_legendre
from readability import Readability
from sklearn.preprocessing import minmax_scale

from my_utils import get_respones_vector
from GUI_Design import ListeningComprehensionApp, data_to_GUI

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
    
    def __init__(self, numberOfMaterials, materialDifficultyLevelNor, questionsDifficultyLevelNor, init_ability=0):
        self.administered_items  = []
        self.selected_items_once = []
        self.selected_all_items = []
        self.response_all = []
        if len(np.array(materialDifficultyLevelNor).shape) == 1:
            self.item_parameter = normalize_item_bank(np.array([materialDifficultyLevelNor]).T)
        else:
            self.item_parameter = normalize_item_bank(np.array(materialDifficultyLevelNor))
        self.ability = init_ability
        self.whole_ability = init_ability
        self.numberOfMaterials = numberOfMaterials
        self.questionsDifficultyLevelNor = questionsDifficultyLevelNor
        self.materialDifficultyLevelNor = materialDifficultyLevelNor

    def normalize_item_parameter(self, one_item_array):
        normalize_item = normalize_item_bank(one_item_array.T)
        return normalize_item
        
    def initial_ability(self):
        # init_ability = RandomInitializer(dist_params=(-2,2)).initialize()
        init_ability = random.uniform(0,4)
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


    def MCMC_ability_estimator(self, response_vector):

        response_once = response_vector

        self.response_all.append(response_once)

        response_once_vec = np.array([response_vector])
        response_all_vec = np.array([response_vector])

        girth_model = GirthMCMC(model='rasch', 
                        options={'n_processors': 4, 'variational_inference': True,'variational_samples': 2000,'n_samples': 2000})

        results_once = girth_model(response_once_vec)
        results_all = girth_model(response_all_vec)

        result_once_diff = results_once['Difficulty']
        result_once_ability = results_once['Ability']

        result_all_diff = results_all['Difficulty']
        result_all_ability = results_all['Ability']

        ability_once = np.average(((result_once_ability)+0.3)*10*(result_once_diff[0]+0.5))
        ability_all = []
        for ability, diff in zip(result_all_ability, result_all_diff):
            ability_all_1d = np.average(((ability)+0.3)*10*(diff+0.5))
            ability_all.append(ability_all_1d)
        ability_all = np.average(ability_all)

        self.ability = ability_once
        self.whole_ability = ability_all

        # if method == 'map':
        #     ability_once = max(np.log(result_once_ability).sum)
        #     ability_all = max(np.log(result_all_ability).sum)
        # elif method == 'eap':
        #     quad_start = -3
        #     quad_stop = 3
        #     quad_int_once = len(result_all_ability)
        #     quad_int_all = len(result_all_ability)

        #     _, w_once = roots_legendre(quad_int_once)
        #     _, w_all = roots_legendre(quad_int_all)

        #     scalar = (quad_stop - quad_start)*0.5
            
        #     weight_once = scalar * w_once
        #     weight_all = scalar * w_all

        #     local_int_once = weight_once * result_once_ability
        #     local_int_all = weight_all * result_all_ability

        #     ability_once = np.sum(local_int_once)/len(local_int_once)
        #     ability_all = np.sum(local_int_all)/len(local_int_all)

        return result_all_diff, ability_once, ability_all

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

    def item_selctor_custmer(self):
        self.selected_items_once = []
        matdiff = self.materialDifficultyLevelNor
        sort_matdiff = sorted(matdiff)
        argsort_matdiff = np.argsort(matdiff)
        
        if self.whole_ability in sort_matdiff:
            diff_same_indices = sort_matdiff.index(self.whole_ability)
            choose_range = argsort_matdiff[diff_same_indices:diff_same_indices+10]
            choose_range = list(choose_range)
            for _ in range(self.numberOfMaterials):
                for used in self.administered_items:
                    if used in choose_range:
                        choose_range.remove(used)
                new_selected_item = random.choice(list(choose_range))
                self.administered_items.append(new_selected_item)
                self.selected_items_once.append(new_selected_item)
                self.selected_all_items.append(new_selected_item)
        else:
            self.item_selector() 
        return self.selected_items_once

# This function will be used to get initial theta and bulit the first object of IRT
def initial_theta(Data, numberOfMaterials, materialDifficultyLevelNor, questionsDifficultyLevelNor):
    irt = Basic_IRT_Adptive_Recommendation(numberOfMaterials=numberOfMaterials, 
                                           materialDifficultyLevelNor=materialDifficultyLevelNor, 
                                           questionsDifficultyLevelNor=questionsDifficultyLevelNor)
    init_ability = irt.initial_ability()
    print("init ability", init_ability)

    select_items = irt.item_selector()
    print("init_select", select_items)
    
    for j in select_items:
        print("Selected Difficulty: ", materialDifficultyLevelNor[j])   
    
    initQuestions, initChoices, initTrueAnswers, initOutputPath = data_to_GUI(Data, select_items)

    print("init true answer", initTrueAnswers) 

    # Driver Code
    app = ListeningComprehensionApp(manyQuestions=initQuestions, manyChoices=initChoices, 
                                    manyTrueAnswers=initTrueAnswers, manyOutputPath=initOutputPath, 
                                    Recommendation=False, LastPart=False)

    selected_answer = app.getAnswer()

    app.mainloop()

    print(selected_answer)

    # Get respones vector
    respones_vector, correct_score= get_respones_vector(selected_answer=selected_answer, true_answer=initTrueAnswers)
    
    recommendation_mode = 'IRT'
    if recommendation_mode == 'IRT':
    
        respones_vector = np.array(respones_vector)
        respones_vector = np.array([[1 if i else 0] for i in respones_vector])

        print("respones vector: ", respones_vector)
        print("Correct score: ", correct_score)
    
        # IRT estimate based on diffculty and discrimination
        new_ability_once, new_ability = irt.IRT_ability_estimator(response_vector=respones_vector)
    elif recommendation_mode == 'MCMC':
        print("Correct score: ", correct_score)
        
        # MCMC estimate based on Bayes Maekvol chain.
        pred_diff, _, new_ability = irt.MCMC_ability_estimator(response_vector=respones_vector)
    
    print(new_ability)
    
    return init_ability, new_ability, irt, select_items, correct_score