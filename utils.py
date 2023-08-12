#!/usr/bin/env python
# coding: utf-8

import irt_parameter_estimation as ipe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.linalg import norm
from Recommendation_Algorithms_Data_Generation import loadMaterialsFeaturesForML

def plot_ICC(abilities, diff, disc, guess):
    ability_range = np.arange(-1, 6, 0.01)
    i=0.01
    probs_correct = []
    row_name = []
    for a,b,c,ability in zip(diff, disc, guess, abilities):
        prob_correct = ipe.logistic3PLabc(a=a, b=b, c=c, theta=ability_range)
        
        intersection_idx = np.argwhere(np.diff(np.sign(ability_range - ability))).flatten()
        
        label = "{} parts test".format(int(i))
        plt.plot(ability_range, prob_correct, label=label)
        i=i+1
        
        #plt.vlines(ability, ymin=0, ymax=1, colors='green', linestyles='--')
        #plt.hlines(prob_correct[intersection_idx], xmin=ability-1.5, xmax=ability+1.5, colors='green', linestyles='--')
        plt.plot(ability_range[intersection_idx], prob_correct[intersection_idx], 'ro')

        row_name.append(label)
        probs_correct.extend(prob_correct[intersection_idx])
        
        #anotation = "prob: {:.1f}".format(prob_correct[intersection_idx][0])
        #plt.annotate(anotation, (ability_range[intersection_idx]-0.5, prob_correct[intersection_idx]), textcoords="offset points",xytext=(0,10),ha="center")
        
        plt.legend()
        plt.xlabel("user ability")
        plt.ylabel("probability of correctly answering")
        
    intersection_df = pd.DataFrame({"ability": abilities, 
                                    "Probs Correcting Answering": probs_correct},index=row_name)  
    display(intersection_df)
    return intersection_df


# Using similarity to evaluate the performance of recommendation algorithms
def get_similarity(selected_items_index, selected_items_answers, reocmmendation_items_index, material_features_path):
    
    material_features = loadMaterialsFeaturesForML(material_features_path)
    
    selected_items = material_features.iloc[selected_items_index]
    recommendation_items = material_features.iloc[reocmmendation_items_index]
    
    selected_items['type'] = ['test']*selected_items.shape[0]
    recommendation_items['type'] = ['recommendation']*len(recommendation_items.index)
    
    selected_items["Answers"] = selected_items_answers
    
    min_distance = []
    max_distance = []
    closest_index = []
    farest_index = []
    for selected_row in selected_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values:
        temp_distance = []
        for recom_row in recommendation_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values:
            temp_distance.append(np.dot(selected_row, recom_row)/(norm(selected_row)*norm(recom_row)))
        min_distance.append(min(temp_distance))
        closest_index.append(recommendation_items['Material Index'].iloc[np.argmin(temp_distance)])
        max_distance.append(max(temp_distance))
        farest_index.append(recommendation_items['Material Index'].iloc[np.argmax(temp_distance)])        

    selected_items["min_similarity"] = min_distance
    selected_items["closest_index"] = closest_index
    selected_items["max_similarity"] = max_distance
    selected_items["farest_index"] = farest_index    
    
    
    min_distance = []
    max_distance = []
    closest_index = []
    farest_index = []
    for recom_row in recommendation_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values:
        temp_distance = []
        for selected_row in selected_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values:
            temp_distance.append(np.dot(selected_row, recom_row)/(norm(selected_row)*norm(recom_row)))
        min_distance.append(min(temp_distance))
        closest_index.append(selected_items['Material Index'].iloc[np.argmin(temp_distance)])
        max_distance.append(max(temp_distance))
        farest_index.append(selected_items['Material Index'].iloc[np.argmax(temp_distance)])         
    recommendation_items["min_similarity"] = min_distance
    recommendation_items["closest_index"] = closest_index    
    recommendation_items["max_similarity"] = max_distance
    recommendation_items["farest_index"] = farest_index    
            
    print(selected_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values)
    recommendation_items[["Materials Difficulty", "Questions Numbers", "Topics"]].values
    
    selected_recomm_items = pd.concat([selected_items, recommendation_items])
    
#     display(selected_items)
#     display(recommendation_items)
    display(selected_recomm_items)


# Define a function to compare selected answers ans true answers, to return a respones vector, which is boolean and return
# the correct scores
def get_respones_vector(selected_answer, true_answer):
    
    respones_vector = []
    correct_score_vector = []
    for i in range(len(true_answer)) :
        temp_score = []
        for j in range(len(true_answer[i])) :
            if len(selected_answer) != 0:
                if true_answer[i][j] == selected_answer[i][j]:
                    respones_vector.append(True)
                    temp_score.append(1)
                else:
                    respones_vector.append(False)
                    temp_score.append(0)
            else:
                respones_vector.append(False)
                temp_score.append(0)    
        correct_score_vector.append(sum(temp_score)/len(temp_score))
    return respones_vector, correct_score_vector


def compute_init_ability(Answers):
    choices_weights = [[10,0],[1,3,5,7],[4,7,6,3],[3,0],[2,4,7,10],[2,3,4,5]]
    score_count = 0
    for a,w in zip(Answers,choices_weights):
        if a == 'A':
            score_count += w[0]
        elif a == 'B':
            score_count += w[1]
        elif a == 'C':
            score_count += w[2]
        elif a == 'D':
            score_count += w[3]
        else:
            score_count += 0
    initial_ability = score_count / 42 * 6
    return initial_ability