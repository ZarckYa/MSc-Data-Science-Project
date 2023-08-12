#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from tkinter import *
from tkinter import messagebox as mb
from DataProcess import Text_To_Speech, read_Questions_and_Choices
from playsound import playsound
from threading import Thread



# Define a class to design a GUI to do the quiz
class Quiz:
    def __init__(self, quizWindow, questions, choices, trueAnswers, mp3Path):
        # Pass Questions, Choices, trueAnswers, mp3Path to class
        self.questions = questions
        self.choices = choices
        self.trueAnswers = trueAnswers
        self.outputPath = mp3Path
        self.quizWindow = quizWindow
        
        # Set the question nunmber and next question number, which will used to locate the questions
        self.questionNum= 0 
        self.questionNextNum = 1
        
        # Set 2 button states, which can make the button disable
        self.stateNextBack = NORMAL
        self.stateAudio = NORMAL
        self.nextOrSubmit = 0
        self.quit = 0
        
        # Audio repeat times used to control audio play time limitation. Correct use to computer the score of quiz.
        # To save all answer user choosen ,it will be used in IRT.
        self.audioRepeatTimes = 1
        self.correct = 0
        self.selectedAnswers = [None]*len(self.questions)

        
        # Those are some functions used to implement GUI
        self.questionStr = StringVar()
        self.ques = self.question(self.questionNum)
        self.detailsExplanation()
        self.opt_selected = StringVar()
        self.opts = self.radiobtns(self.questionNum)
        self.display_options(self.questionNum)
        self.buttons()


    # This function used to dispaly questions one by one.
    def question(self, questionNum):
        # Set a title
        title = Label(self.quizWindow, text="Quiz in Listening Comprehension", width=50, bg="blue", fg="white", font=("times", 20, "bold"))
        title.place(x=0, y=2)
       
        # Use the question string variale to store different questions.
        # Then use the Label function to display it in GUI
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[questionNum])
        questionText = Label(self.quizWindow, textvariable = self.questionStr, width=60, font=("times", 16, "bold"), anchor="w")
        questionText.place(x=70, y=150)
        return questionText

    # This function used to dispaly some explanation of Questions, 
    # which conrtains number of questions and the tips for audio play times limitation
    def detailsExplanation(self):
        
        # Tip for number of questions
        quesNumTips = "In this Audio, will involved " + str(len(self.questions)) + " questions."
        questionNumTips = Label(self.quizWindow, text = quesNumTips, fg='blue', width=60, font=("times", 16, "bold"), anchor="w")
        questionNumTips.place(x=70, y=120)
        
        # Tip for audio play times limitation
        audNumTips = "Each audio, just \n can be play twice"
        audioNumTips = Label(self.quizWindow, text = audNumTips, width=14, bg = "red", font=("times", 16, "bold"), anchor="w")
        audioNumTips.place(x=600, y=200)         
    
    # This funcrion use to play the mp3 file
    def audio(self):
        # play mp3 sound
        playsound(self.outputPath)
        
        # Set a play times count, if over 2times, the audio play bitton will disable
        self.audioRepeatTimes += 1
        if self.audioRepeatTimes == 3:
            self.stateAudio = DISABLED
            self.buttons()
    
    # To keep other button still work when the audio is playing, so set a backgroud.
    def audioBackgroug(self):
        t = Thread(target=self.audio)
        t.start()
    
    # This function will be used to set some choices. There are A, B, C three choices
    def radiobtns(self, questionNum):
        b = []
        yp = 200
        if len(self.choices[questionNum]) == 3:
            for val in ['A', 'B', 'C']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40
        elif len(self.choices[questionNum]) == 4:
            for val in ['A', 'B', 'C', 'D']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40            
        return b
    
    # Show all choices content
    def display_options(self, questionNum):
        
        # Through the index of questions to show all chouices corresponding to this questions
        val = 0
        self.opt_selected.set(0)
        self.ques['text'] = self.questions[questionNum]
        for op in self.choices[questionNum]:
            self.opts[val]['text'] = op
            val += 1

    # Destroy all exited radio button        
    def clear_options(self, questionNum):
        # Through the index of questions to show all chouices corresponding to this questions
        val = 0
        for op in self.choices[questionNum]:
            self.opts[val].destroy()
            val += 1

    # Set 4 buttons to run the GUI, stateNextBack use to control the Back and Next button available or not, 
    # stateAudio use to control audio play button
    def buttons(self):
        # This is Next button or submit button
        if len(self.questions) == 1:
            self.nextOrSubmit = 1
            
        # A button state controled by the nextOrSubmit variable.
        # When you do the last question, the next botton will become submit.
        if self.nextOrSubmit == 0:
            nextButton = Button(self.quizWindow, text="Next",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack
        elif self.nextOrSubmit == 1:
            nextButton = Button(self.quizWindow, text="Submit Section",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack           
        
        # Back button
        backButton = Button(self.quizWindow, text="Back",command=self.backbtn, width=10,bg="green",fg="white",font=("times",16,"bold"))
        backButton.place(x=20,y=380)
        backButton['state'] = self.stateNextBack

        # Quit button
        quitButton = Button(self.quizWindow, text="Quit Section", command=self.quitTest,width=10,bg="red",fg="white", font=("times",16,"bold"))
        quitButton.place(x=380,y=380)
        
        #Audio play button
        audioButton = Button(self.quizWindow, text="Audio", command=self.audioBackgroug,width=10,bg="blue",fg="white", font=("times",16,"bold"))
        audioButton.place(x=620,y=70)        
        audioButton['state'] = self.stateAudio
    
    # Save all choices made by tester
    def saveAns(self, questionNum):
        self.selectedAnswers[questionNum] = self.opt_selected.get()
    
    # Set a quit signal
    def quitTest(self):
        self.quit = 1
        self.quizWindow.destroy()
    
    # Next botton function, it will change the index number of questions, so it can turn into next questions.
    # If the questions is the last one.
    def nextbtn(self):
        # Save all answera made by user.
        # Change the questionn index
        self.saveAns(self.questionNum)
        self.questionNum += 1
        self.questionNextNum += 1
        
        # if this question is last one, so the next button will become submit
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
        # if tester submit quiz, Next and Back botton will disable and return the score the user get.
        # If not last question, go into next question
        if self.questionNum == len(self.questions):
            self.display_result()
            self.stateNextBack = DISABLED
            self.stateAudio = DISABLED
            self.buttons()
        else:
            self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
            self.clear_options(self.questionNum - 1)
            self.opts = self.radiobtns(self.questionNum)
            self.display_options(self.questionNum)       
    
    # Function use to go back to last question
    def backbtn(self):
        # This will chech the question number if is begger than 1. If the question is first question., 
        # the back button can clike but not work
        if self.questionNextNum > 1:
            self.questionNum -= 1
            self.questionNextNum -= 1
        # This will change the sSubmit button to Next botton            
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
            
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
        self.clear_options(self.questionNum + 1)
        self.opts = self.radiobtns(self.questionNum)
        self.display_options(self.questionNum)         
    
    # Show the score of user   
    def display_result(self):
        correctIndex = []
        wrongIndex = []
        # Check all answer user made and check it right or not
        for i, answer in enumerate(self.selectedAnswers):
            if answer == self.trueAnswers[i]:
                self.correct += 1
                correctIndex.append(i+1)
            else:
                wrongIndex.append(i+1)
        # Compute the score in percentage
        score = int(self.correct / len(self.questions) * 100)
        result = "Score: " + str(score) + "%"
        # To show the right answer and wrong answer
        wc = len(self.questions) - self.correct
        correct = "No. of correct answers: " + str(self.correct)
        wrong = "No. of wrong answers: " + str(wc)
        # To show the index of right and wrong answer.
        correctIndices = "The index of correct answer: " + str(correctIndex)
        wrongIndices = "The index of wrong answer: " + str(wrongIndex)
        
        mb.showinfo("Result", "\n".join([result, correct, wrong, correctIndices, wrongIndices]))
        return self.selectedAnswers



# Define a class to design a GUI to do the quiz
class Recommendation_Questions:
    def __init__(self, quizWindow, questions, choices, trueAnswers, mp3Path):
        # Pass Questions, Choices, trueAnswers, mp3Path to class
        self.questions = questions
        self.choices = choices
        self.trueAnswers = trueAnswers
        self.outputPath = mp3Path
        self.quizWindow = quizWindow
        
        # Set the question nunmber and next question number, which will used to locate the questions
        self.questionNum= 0 
        self.questionNextNum = 1
        
        # Set 2 button states, which can make the button disable
        self.stateNextBack = NORMAL
        self.stateAudio = NORMAL
        self.nextOrSubmit = 0
        self.quit = 0
        
        # Audio repeat times used to control audio play time limitation. Correct use to computer the score of quiz.
        # To save all answer user choosen ,it will be used in IRT.
        self.correct = 0
        self.selectedAnswers = [None]*len(self.questions)
        self.display = False
        
        # Those are some functions used to implement GUI
        self.questionStr = StringVar()
        self.trueAnswerStr = StringVar()
        self.ques = self.question(self.questionNum)
        self.detailsExplanation()
        self.opt_selected = StringVar()
        self.opts = self.radiobtns(self.questionNum)
        self.display_options(self.questionNum)
        self.buttons()
        self.display_correct_options()


    # This function used to dispaly questions one by one.
    def question(self, questionNum):
        # Set a title
        title = Label(self.quizWindow, text="Reocmmendation of Listening Comprehension Questions", 
                      width=50, bg="blue", fg="white", font=("times", 20, "bold"))
        title.place(x=0, y=2)
       
        # Use the question string variale to store different questions.
        # Then use the Label function to display it in GUI
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[questionNum])
        questionText = Label(self.quizWindow, textvariable = self.questionStr, width=60, font=("times", 16, "bold"), anchor="w")
        questionText.place(x=70, y=150)
        return questionText

    # This function used to dispaly some explanation of Questions, 
    # which conrtains number of questions and the tips for audio play times limitation
    def detailsExplanation(self):
        
        # Tip for number of questions
        quesNumTips = "In this Audio, will involved " + str(len(self.questions)) + " questions."
        questionNumTips = Label(self.quizWindow, text = quesNumTips, fg='blue', width=60, font=("times", 16, "bold"), anchor="w")
        questionNumTips.place(x=70, y=120)
        
#         # Tip for audio play times limitation
#         audNumTips = "Each audio, just \n can be play twice"
#         audioNumTips = Label(self.quizWindow, text = audNumTips, width=14, bg = "red", font=("times", 16, "bold"), anchor="w")
#         audioNumTips.place(x=600, y=200)         
    
    # This funcrion use to play the mp3 file
    def audio(self):
        # play mp3 sound
        playsound(self.outputPath)
        
        # Set a play times count, if over 2times, the audio play bitton will disable
#         self.audioRepeatTimes += 1
#         if self.audioRepeatTimes == 3:
#             self.stateAudio = DISABLED
#             self.buttons()
    
    # To keep other button still work when the audio is playing, so set a backgroud.
    def audioBackgroug(self):
        t = Thread(target=self.audio)
        t.start()
    
    # This function will be used to set some choices. There are A, B, C three choices
    def radiobtns(self, questionNum):
        b = []
        yp = 200
        if len(self.choices[questionNum]) == 3:
            for val in ['A', 'B', 'C']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40
        elif len(self.choices[questionNum]) == 4:
            for val in ['A', 'B', 'C', 'D']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40            
        return b
    
    # Show all choices content
    def display_options(self, questionNum):
        
        # Through the index of questions to show all chouices corresponding to this questions
        val = 0
        self.opt_selected.set(0)
        self.ques['text'] = self.questions[questionNum]
        for op in self.choices[questionNum]:
            self.opts[val]['text'] = op
            val += 1
    
    # Destroy all exited radio button        
    def clear_options(self, questionNum):
        # Through the index of questions to show all chouices corresponding to this questions
        val = 0
        for op in self.choices[questionNum]:
            self.opts[val].destroy()
            val += 1


    def display_correct_options(self):
        
        if self.display:
            self.trueAnswerStr.set("True Answer: "+self.trueAnswers[self.questionNum])
        else:
            self.trueAnswerStr.set("")
        questionText = Label(self.quizWindow, textvariable = self.trueAnswerStr, fg='green', width=60, font=("times", 16, "bold"), anchor="w")
        questionText.place(x=620, y=250)   
    
    # Set 4 buttons to run the GUI, stateNextBack use to control the Back and Next button available or not, 
    # stateAudio use to control audio play button
    def buttons(self):
        # This is Next button or submit button
        if len(self.questions) == 1:
            self.nextOrSubmit = 1
            
        # A button state controled by the nextOrSubmit variable.
        # When you do the last question, the next botton will become submit.
        if self.nextOrSubmit == 0:
            nextButton = Button(self.quizWindow, text="Next",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack
        elif self.nextOrSubmit == 1:
            nextButton = Button(self.quizWindow, text="Submit Section",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack           
        
        # Back button
        backButton = Button(self.quizWindow, text="Back",command=self.backbtn, width=10,bg="green",fg="white",font=("times",16,"bold"))
        backButton.place(x=20,y=380)
        backButton['state'] = self.stateNextBack

        # Quit button
        quitButton = Button(self.quizWindow, text="Quit Section", command=self.quitTest,width=10,bg="red",fg="white", font=("times",16,"bold"))
        quitButton.place(x=380,y=380)
        
        # Audio play button
        audioButton = Button(self.quizWindow, text="Audio", command=self.audioBackgroug,width=10,bg="blue",fg="white", font=("times",16,"bold"))
        audioButton.place(x=620,y=70)  
        
        audioButton['state'] = self.stateAudio
        
        
        # Show the correct answers
        if self.display == False:
            answerButton = Button(self.quizWindow, text="Show Answer", command=self.display_correct_options_btn,width=10,bg="pink",fg="white", font=("times",16,"bold"))
            answerButton.place(x=620,y=200)         
        elif self.display == True:
            answerButton = Button(self.quizWindow, text="Hide Answer", command=self.display_correct_options_btn,width=10,bg="pink",fg="white", font=("times",16,"bold"))
            answerButton.place(x=620,y=200)             
        answerButton['state'] = self.stateNextBack
        
    # Save all choices made by tester
    def saveAns(self, questionNum):
        self.selectedAnswers[questionNum] = self.opt_selected.get()
    
    # Button to control the right answer showuld be answerws or not.
    def display_correct_options_btn(self):
        if self.display == False:
            self.display = True
            self.display_correct_options()
        else:
            self.display = False
            self.display_correct_options()
        
        self.buttons()
            
    # Set a quit signal
    def quitTest(self):
        self.quit = 1
        self.quizWindow.destroy()
    
    # Next botton function, it will change the index number of questions, so it can turn into next questions.
    # If the questions is the last one.
    def nextbtn(self):
        # Save all answera made by user.
        # Change the questionn index
        self.saveAns(self.questionNum)
        self.questionNum += 1
        self.questionNextNum += 1
        
        # if this question is last one, so the next button will become submit
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
        # if tester submit quiz, Next and Back botton will disable and return the score the user get.
        # If not last question, go into next question
        if self.questionNum == len(self.questions):
            self.display_result()
            self.stateNextBack = DISABLED
            self.buttons()
        else:
            self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
            self.clear_options(self.questionNum - 1)
            self.opts = self.radiobtns(self.questionNum)
            self.display_options(self.questionNum)
            self.display = False
            self.display_correct_options()
    
    # Function use to go back to last question
    def backbtn(self):
        # This will chech the question number if is begger than 1. If the question is first question., 
        # the back button can clike but not work
        if self.questionNextNum > 1:
            self.questionNum -= 1
            self.questionNextNum -= 1
        # This will change the sSubmit button to Next botton            
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
            
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
        self.clear_options(self.questionNum + 1)
        self.opts = self.radiobtns(self.questionNum)
        self.display_options(self.questionNum)   
        self.display = False
        self.display_correct_options()
        
    # Show the score of user   
    def display_result(self):
        correctIndex = []
        wrongIndex = []
        # Check all answer user made and check it right or not
        for i, answer in enumerate(self.selectedAnswers):
            if answer == self.trueAnswers[i]:
                self.correct += 1
                correctIndex.append(i+1)
            else:
                wrongIndex.append(i+1)
        # Compute the score in percentage
        score = int(self.correct / len(self.questions) * 100)
        result = "Score: " + str(score) + "%"
        # To show the right answer and wrong answer
        wc = len(self.questions) - self.correct
        correct = "No. of correct answers: " + str(self.correct)
        wrong = "No. of wrong answers: " + str(wc)
        # To show the index of right and wrong answer.
        correctIndices = "The index of correct answer: " + str(correctIndex)
        wrongIndices = "The index of wrong answer: " + str(wrongIndex)
        
        mb.showinfo("Result", "\n".join([result, correct, wrong, correctIndices, wrongIndices]))
        return self.selectedAnswers

class Initial_Questionnaire:
    def __init__(self, quizWindow, questions, choices):
        # Pass Questions, Choices, trueAnswers, mp3Path to class
        self.questions = questions
        self.choices = choices
        self.quizWindow = quizWindow
        
        # Set the question nunmber and next question number, which will used to locate the questions
        self.questionNum= 0 
        self.questionNextNum = 1
        
        # Set 2 button states, which can make the button disable
        self.stateNextBack = NORMAL
        self.nextOrSubmit = 0
        self.quit = 0
        
        # Audio repeat times used to control audio play time limitation. Correct use to computer the score of quiz.
        # To save all answer user choosen ,it will be used in IRT.
        self.correct = 0
        self.selectedAnswers = [None]*len(self.questions)
        self.display = False
        
        # Those are some functions used to implement GUI
        self.questionStr = StringVar()
        self.trueAnswerStr = StringVar()
        self.ques = self.question(self.questionNum)
        self.detailsExplanation()
        self.opt_selected = StringVar()
        self.opts = self.radiobtns(self.questionNum)
        self.display_options(self.questionNum)
        self.buttons()

    # This function used to dispaly questions one by one.
    def question(self, questionNum):
        # Set a title
        title = Label(self.quizWindow, text="Initial Ability Estimation Questionnaire", 
                      width=50, bg="blue", fg="white", font=("times", 20, "bold"))
        title.place(x=0, y=2)
       
        # Use the question string variale to store different questions.
        # Then use the Label function to display it in GUI
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[questionNum])
        questionText = Label(self.quizWindow, textvariable = self.questionStr, width=60, font=("times", 16, "bold"), anchor="w")
        questionText.place(x=70, y=150)
        return questionText

    # This function used to dispaly some explanation of Questions, 
    # which conrtains number of questions and the tips for audio play times limitation
    def detailsExplanation(self):
        
        # Tip for number of questions
        quesNumTips = "In this questionnaire, will involved " + str(len(self.questions)) + " questions."
        questionNumTips = Label(self.quizWindow, text = quesNumTips, fg='blue', width=60, font=("times", 15, "bold"), anchor="w")
        questionNumTips.place(x=70, y=120)

        # Tip for functions
        funcTips1 = "This is a questionnaire to estimate your ability, Please answer the questions based on your real situation."
        funcTips2 = "If this phase accurately assesses your abilities and preferences,it will be possible to provide more"
        funcTips3 = "accurate recommendations and also reduce the follow test phaseand shorten the testing process."

        functionsTips1 = Label(self.quizWindow, text = funcTips1, fg='green', width=100, font=("times", 12), anchor="w")
        functionsTips2 = Label(self.quizWindow, text = funcTips2, fg='green', width=100, font=("times", 12), anchor="w")
        functionsTips3 = Label(self.quizWindow, text = funcTips3, fg='green', width=100, font=("times", 12), anchor="w")
        
        functionsTips1.place(x=70, y=50)
        functionsTips2.place(x=70, y=70)
        functionsTips3.place(x=70, y=90)      

    
    # This function will be used to set some choices. There are A, B, C three choices
    def radiobtns(self, questionNum):
        b = []
        yp = 200
        if len(self.choices[questionNum]) == 3:
            for val in ['A', 'B', 'C']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40
        elif len(self.choices[questionNum]) == 4:
            for val in ['A', 'B', 'C', 'D']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40
        elif len(self.choices[questionNum]) == 2:
            for val in ['A', 'B']:
                btn = Radiobutton(self.quizWindow, text=" ", variable=self.opt_selected, value=val, font=("times", 14))
                b.append(btn)
                btn.place(x=100, y=yp)
                yp += 40
        return b
    
    # Show all choices content
    def display_options(self, questionNum):
         
        val = 0 
        self.opt_selected.set(0)
        self.ques['text'] = self.questions[questionNum]
        for op in self.choices[questionNum]:
            self.opts[val]['text'] = op
            val += 1  

    # Destroy all exited radio button        
    def clear_options(self, questionNum):
        # Through the index of questions to show all chouices corresponding to this questions
        val = 0
        for op in self.choices[questionNum]:
            self.opts[val].destroy()
            val += 1
        
    # Set 4 buttons to run the GUI, stateNextBack use to control the Back and Next button available or not, 
    # stateAudio use to control audio play button
    def buttons(self):
        # This is Next button or submit button
        if len(self.questions) == 1:
            self.nextOrSubmit = 1
            
        # A button state controled by the nextOrSubmit variable.
        # When you do the last question, the next botton will become submit.
        if self.nextOrSubmit == 0:
            nextButton = Button(self.quizWindow, text="Next",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack
        elif self.nextOrSubmit == 1:
            nextButton = Button(self.quizWindow, text="Submit Section",command=self.nextbtn, width=12,bg="green",fg="white",font=("times",16,"bold"))
            nextButton.place(x=200,y=380)
            nextButton['state'] = self.stateNextBack           
        
        # Back button
        backButton = Button(self.quizWindow, text="Back",command=self.backbtn, width=10,bg="green",fg="white",font=("times",16,"bold"))
        backButton.place(x=20,y=380)
        backButton['state'] = self.stateNextBack

        # Quit button
        quitButton = Button(self.quizWindow, text="Quit Section", command=self.quitTest,width=10,bg="red",fg="white", font=("times",16,"bold"))
        quitButton.place(x=380,y=380)
         
        
    # Save all choices made by tester
    def saveAns(self, questionNum):
        self.selectedAnswers[questionNum] = self.opt_selected.get()
            
    # Set a quit signal
    def quitTest(self):
        self.quit = 1
        self.quizWindow.destroy()
    
    # Next botton function, it will change the index number of questions, so it can turn into next questions.
    # If the questions is the last one.
    def nextbtn(self):
        # Save all answera made by user.
        # Change the questionn index
        self.saveAns(self.questionNum)
        self.questionNum += 1
        self.questionNextNum += 1
        
        # if this question is last one, so the next button will become submit
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
        # if tester submit quiz, Next and Back botton will disable and return the score the user get.
        # If not last question, go into next question
        if self.questionNum == len(self.questions):
            self.stateNextBack = DISABLED
            self.buttons()
        else:
            self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
            self.clear_options(self.questionNum - 1)
            self.opts = self.radiobtns(self.questionNum)
            self.display_options(self.questionNum)
            self.display = False
    
    # Function use to go back to last question
    def backbtn(self):
        # This will chech the question number if is begger than 1. If the question is first question., 
        # the back button can clike but not work
        if self.questionNextNum > 1:
            self.questionNum -= 1
            self.questionNextNum -= 1
        # This will change the sSubmit button to Next botton            
        if self.questionNum >= len(self.questions) - 1:
            self.nextOrSubmit = 1
            self.buttons()
        else:
            self.nextOrSubmit = 0
            self.buttons()
            
        self.questionStr.set(str(self.questionNextNum)+". "+self.questions[self.questionNum])
        self.clear_options(self.questionNum + 1)
        self.opts = self.radiobtns(self.questionNum) 
        self.display_options(self.questionNum)  
        self.display = False  

# To design a function to transfer the all materials text into audio and questions.
# The audio will be show in a GUI to provide it to user
def data_to_GUI(Data, indices):
    
    manyQuestions = []
    manyChoices = []
    manyTrueAnswers = []
    manyOutputPath = []
    
    # Go through all indices of materials and do transfer operation
    for i in indices:
        # Set a output path
        outputPath = "Data/Conversation mp3/index" + str(i) + ".mp3"
        oneData = Data[i]
        # Call the functions from another python file and split questions, choices, answers.
        questions, choices, trueAnswers = read_Questions_and_Choices(oneData)
        manyQuestions.append(questions)
        manyChoices.append(choices)
        manyTrueAnswers.append(trueAnswers)
        manyOutputPath.append(outputPath)
        # Convert text into speech
        Text_To_Speech(oneData, outputPath)
    
    return  manyQuestions, manyChoices, manyTrueAnswers, manyOutputPath


class ListeningComprehensionApp(Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, manyQuestions, manyChoices, manyTrueAnswers, manyOutputPath, Recommendation, LastPart, *args, **kwargs):
        
        # __init__ function for class Tk
        Tk.__init__(self, *args, **kwargs)

        self.title("Listening Compresension Quiz")
        # creating a container
        container = Frame(self, width=800, height=500)
        container.pack(fill = None, expand = False)

        # Set min frame size
        container.grid_rowconfigure(0, minsize=500, weight = 1)
        container.grid_columnconfigure(0, minsize=800, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
        self.frameIndex = 0
        self.resultCont = container
        
        self.manyQuestions = manyQuestions
        self.manyChoices = manyChoices
        self.manyTrueAnswers = manyTrueAnswers
        self.manyOutputPath = manyOutputPath
        
        self.materialLen = len(self.manyOutputPath)
        self.totalQuesNum = sum(len(q) for q in self.manyQuestions)
        
        self.answers = []
        
        self.buttonControl = 0
        self.Index = 0
        self.lastpart = LastPart
        # iterating through a tuple consisting
        # of the different page layouts
        for i in range(len(self.manyQuestions)):
            
            if i == 0:
                self.buttonControl = 1
            elif i == (len(self.manyQuestions) - 1):
                self.buttonControl = 2
            else:
                self.buttonControl = 0
                
            self.frameIndex = i+1
            if Recommendation == False:
                frame = Materials(container, self, self.manyQuestions[i], self.manyChoices[i], 
                                  self.manyTrueAnswers[i], self.manyOutputPath[i])
            else:
                frame = RecommendationMaterials(container, self, self.manyQuestions[i], self.manyChoices[i], 
                                  self.manyTrueAnswers[i], self.manyOutputPath[i])                
                
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[i] = frame

#             if i == len(manyQuestions) - 1:
#                 frame = Results(container, self, manyTrueAnswers)
#                 self.frames[i+1] = frame
            
            frame.grid(row = 0, column = 0, sticky ="nsew")
        
        self.show_frame(index=0)

    # to display the current frame passed as
    # parameter
    def show_frame(self, index):
        frame = self.frames[index]
        frame.tkraise()
    
    def destroy_frames(self):
#         for index in range(len(manyQuestions)):
#             self.answers.append(self.frames[index].oneSegmentAnswer)
#         print(self.answers)
        self.destroy()
    
    def submit_frames(self):
        for index in range(len(self.manyQuestions)):
            self.answers.append(self.frames[index].oneSegmentAnswer)
            self.frames[index].destroy()
            
#         print(self.answers)             
        frame = Results(parent = self.resultCont, controller = self, TrueAnswers = self.manyTrueAnswers, selected_answers = self.answers)
        self.frames[len(self.manyQuestions)] = frame
            
        frame.grid(row = 0, column = 0, sticky ="nsew")
        self.show_frame(index=len(self.manyQuestions))    
    
    def indexDown(self):
        self.Index -= 1

    def indexUp(self):
        self.Index += 1
    
    def getAnswer(self):
        return self.answers

# To construct the Materials Frame, which will display in GUI, this will support user to answer questions.    
class Materials(Frame):
    
    def __init__(self, parent, controller, oneQuestions, oneChoices, oneTrueAnswers, oneOutputPath):
        Frame.__init__(self, parent)
        
        # Show some Tips
        textTips1 = "In this Quiz, you will finish " + str(controller.materialLen) + " listening materials, \n" + "you have total " + str(controller.totalQuesNum) + " questions"
        
        text1 = Label(self, text = textTips1, width=40, fg='blue', font=("times", 16, "bold"), anchor="w")
        text1.place(x=70, y=50)
        
        textTips2 = "A." + str(controller.frameIndex) + ":"
        
        text2 = Label(self, text = textTips2, width=40, fg='green', font=("times", 16, "bold"), anchor="w")
        text2.place(x=27, y=120)
        
        oneMaterial = Quiz(quizWindow=self, questions=oneQuestions, choices=oneChoices, 
                           trueAnswers=oneTrueAnswers, mp3Path=oneOutputPath)
        
        self.oneSegmentAnswer = oneMaterial.selectedAnswers
        self.quizFlag = oneMaterial.quit
        
        button1 = Button(self, text ="Back Material", 
                         command = lambda : [controller.show_frame(controller.Index - 1), controller.indexDown()],
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
     
        # putting the button in its place by
        # using grid
        button1.place(x=620,y=300)
        
        if controller.buttonControl == 1:
            button1.config(state=DISABLED)

        ## button to show frame 2 with text layout2
        button2 = Button(self, text ="Next Material", 
                         command = lambda : [controller.show_frame(controller.Index + 1), controller.indexUp()],
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
     
        # putting the button in its place by
        # using grid
        button2.place(x=620,y=350)

        if controller.buttonControl == 2:
            button2.config(state=DISABLED)
        
        if controller.lastpart:
            button3 = Button(self, text ="Quit Quiz", 
                             command = lambda : controller.destroy_frames(),
                             width=10,bg="red",fg="white",font=("times",16,"bold"))

            button3.place(x=620,y=400)
        else:
            button3 = Button(self, text ="Quit Part", 
                             command = lambda : controller.destroy_frames(),
                             width=10,bg="red",fg="white",font=("times",16,"bold"))

            button3.place(x=620,y=400)            
        
        button4 = Button(self, text ="Submit Quiz", 
                         command = lambda : controller.submit_frames(),
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
        
        button4.place(x=620,y=450)

        
class Results(Frame):
    
    def __init__(self, parent, controller, TrueAnswers, selected_answers):
        Frame.__init__(self, parent)
        
        title = Label(self, text="Listening Comprehension Result", width=50, bg="blue", fg="white", font=("times", 20, "bold"))
        title.place(x=0, y=2)
        
        Text1 = Label(self, text = "Quiz Result:", width=60, font=("times", 20, "bold"), anchor="w")
        Text1.place(x=30, y=50)        
        
        TextStr2 = "You total test " + str(len(selected_answers)) + " Materials"
        Text2 = Label(self, text = TextStr2, width=60, font=("times", 18, "bold"), anchor="w")
        Text2.place(x=30, y=100)         
        
        self.correctAnswers = [[] for i in range(len(TrueAnswers))]

        print("selected: ", selected_answers)
        print("True: ", TrueAnswers)
        print("selected len: ", len(selected_answers))
        for i in range(len(selected_answers)) :
            for j in range(len(selected_answers[i])) :
                if TrueAnswers[i][j] == selected_answers[i][j]:
                    self.correctAnswers[i].append(1)

                else:
                    self.correctAnswers[i].append(0)

        yLocTextStr3 = 140
        yLocTextStr4 = 180
        for k in range(len(TrueAnswers)):
            textStr3 = "Material " + str(k+1) + " Mark: " + str(sum(self.correctAnswers[k])/len(self.correctAnswers[k]) * 100) + "%"
            Text2 = Label(self, text = textStr3, width=60, font=("times", 16, "bold"), anchor="w")
            Text2.place(x=30, y=yLocTextStr3)
            yLocTextStr3 = yLocTextStr4 + 40
            
            textStr4 = "The answer details: " + str(self.correctAnswers[k])
            Text3 = Label(self, text = textStr4, width=60, font=("times", 16, "bold"), anchor="w")
            Text3.place(x=30, y=yLocTextStr4)
            yLocTextStr4 = yLocTextStr3 + 40            
            
        if controller.lastpart:
            button3 = Button(self, text ="Quit Quiz", 
                             command = lambda : controller.destroy_frames(),
                             width=10,bg="red",fg="white",font=("times",16,"bold"))

            button3.place(x=620,y=400)
        else:
            button3 = Button(self, text ="Next Part", 
                             command = lambda : controller.destroy_frames(),
                             width=10,bg="red",fg="white",font=("times",16,"bold"))

            button3.place(x=620,y=400)     
        
class RecommendationMaterials(Frame):
    
    def __init__(self, parent, controller, oneQuestions, oneChoices, oneTrueAnswers, oneOutputPath):
        Frame.__init__(self, parent)

        textTips1 = "In this recommendation phase, you can go through \n" + str(controller.materialLen) + " listening materials, " + "you have total " + str(controller.totalQuesNum) + " questions"
        
        text1 = Label(self, text = textTips1, width=40, fg='blue', font=("times", 16, "bold"), anchor="w")
        text1.place(x=70, y=50)

        textTips2 = "A." + str(controller.frameIndex) + ":"
        
        text2 = Label(self, text = textTips2, width=40, fg='green', font=("times", 16, "bold"), anchor="w")
        text2.place(x=27, y=120)
        
        oneMaterial = Recommendation_Questions(quizWindow=self, questions=oneQuestions, choices=oneChoices, 
                                               trueAnswers=oneTrueAnswers, mp3Path=oneOutputPath)
        
        self.oneSegmentAnswer = oneMaterial.selectedAnswers
        self.quizFlag = oneMaterial.quit
        
        button1 = Button(self, text ="Back Material", 
                         command = lambda : [controller.show_frame(controller.Index - 1), controller.indexDown()],
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
     
        # putting the button in its place by
        # using grid
        button1.place(x=620,y=300)
        
        if controller.buttonControl == 1:
            button1.config(state=DISABLED)

        ## button to show frame 2 with text layout2
        button2 = Button(self, text ="Next Material", 
                         command = lambda : [controller.show_frame(controller.Index + 1), controller.indexUp()],
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
     
        # putting the button in its place by
        # using grid
        button2.place(x=620,y=350)

        if controller.buttonControl == 2:
            button2.config(state=DISABLED)
        
        button3 = Button(self, text ="Quit Recom", 
                         command = lambda : controller.destroy_frames(),
                         width=10,bg="red",fg="white",font=("times",16,"bold"))
        
        button3.place(x=620,y=400)
    
        
        button4 = Button(self, text ="Submit Quiz", 
                         command = lambda : controller.submit_frames(),
                         width=10,bg="green",fg="white",font=("times",16,"bold"))
        
        button4.place(x=620,y=450)
    
