#!/usr/bin/env python
# coding: utf-8



import json
import numpy as np
import re
import pyttsx3
from moviepy.editor import concatenate_audioclips, AudioFileClip
from playsound import playsound


def Text_To_Speech(oneData, outputPath):
    # Read the conversation from dataset
    singleConversation = list(oneData.keys())[0]
    # Clean data that any string contains "(number)", like (4), (13)
    cleanedConversationText = re.sub(r'\(*\d*\)', r'',singleConversation)
    
    # Set the language to English
    # Split data by "---", which conversation begining as "---"
    Language = 'en'
    roles = cleanedConversationText.split("---")

    # define a empty list to store all mp3 segment path, 
    # which will used to concat into a whole conversation with man and women voices
    mp3_roles = []

    # The loop used to seperate text said by man or women
    for i, role in enumerate(roles):
        if len(role) != 0:
            if "W" == role[0]:
                role = role[3:]
                speaker = pyttsx3.init()
                voices = speaker.getProperty("voices")
                speaker.setProperty('voice', voices[1].id)
                path = "Data/Conversation mp3/Temp" + str(i) + ".mp3"
                speaker.save_to_file(role, path)
                speaker.runAndWait()
                mp3_roles.append(path)
            elif (role[0] != "W") & (role[0] != "M"):
                role = role[3:]
                speaker = pyttsx3.init()
                voices = speaker.getProperty("voices")
                speaker.setProperty('voice', voices[1].id)
                path = "Data/Conversation mp3/Temp" + str(i) + ".mp3"
                speaker.save_to_file(role, path)
                speaker.runAndWait()
                mp3_roles.append(path)            
            elif "M" == role[0]:
                role = role[3:]
                speaker = pyttsx3.init()
                voices = speaker.getProperty("voices")
                speaker.setProperty('voice', voices[2].id)
                path = "Data/Conversation mp3/Temp" + str(i) + ".mp3"
                speaker.save_to_file(role, path)
                speaker.runAndWait()
                mp3_roles.append(path)
            elif (role[0] != "W") & (role[0] != "M"):
                role = role[3:]
                speaker = pyttsx3.init()
                voices = speaker.getProperty("voices")
                speaker.setProperty('voice', voices[2].id)
                path = "Data/Conversation mp3/Temp" + str(i) + ".mp3"
                speaker.save_to_file(role, path)
                speaker.runAndWait()  
    # Read all map3 segments from local
    clips = [AudioFileClip(c) for c in mp3_roles]
    # Concat all segment into one conversation
    final_clip = concatenate_audioclips(clips)
    # Output and save it into local
    final_clip.write_audiofile(outputPath)         



def read_Questions_and_Choices(oneData):
    
    # Define 3 lists to store all questions, choices and answers
    questions = []
    choices = []
    trueAnswers = []
    # Read questions and choices
    questionsWithAnswers = list(oneData.values())[0]
    
    # This loop use to go through all questions in the whole dict
    for questionWithAnswer in questionsWithAnswers:
        # To get the question from dict
        question = list(questionWithAnswer.keys())[0]
        questions.append(question)
        
        # To get the choices from dict and split choices by "***"
        choiceList = list(list(questionWithAnswer.values())[0].keys())[0]
        choice = choiceList.split('***')
        choices.append(choice)
        
        # To get all true answers in dict
        trueAnswer = list(list(questionWithAnswer.values())[0].values())[0]
        trueAnswers.append(trueAnswer)
    return questions, choices, trueAnswers
