#!/usr/bin/env python
# coding: utf-8

# In[21]:


import argparse
import datetime
import logging
import numpy as np
import tensorflow as tf
import os
from load_data import DataLoader
from model import DeepIRTModel
from run import run_model
from utils import getLogger
from configs import ModelConfigFactory

# set logger
logger = getLogger('Deep-IRT-model')

# _args = parser.parse_args()
# args = ModelConfigFactory.create_model_config(_args)
# logger.info("Model Config: {}".format(args))

class argus:
    def __init__(self):
        # training setting
        self.dataset = 'my_data'
        self.save= False
        self.cpu= True
        self.n_epochs= 1
        self.batch_size= 1
        self.train= True
        self.show= True
        self.learning_rate= 0.003
        self.max_grad_norm= 10.0
        self.use_ogive_model= False
        # dataset param
        self.seq_len= 8
        self.n_questions= 12443
        self.data_dir= './DeepIRT/data/mydata'
        self.data_name= 'my_data'
        # DKVMN param
        self.memory_size= 10
        self.key_memory_state_dim= 10
        self.value_memory_state_dim= 20
        self.summary_vector_output_dim= 10
        # parameter for the SA Network and KCD network
        self.student_ability_layer_structure= None
        self.question_difficulty_layer_structure= None
        self.discimination_power_layer_structure= None
        # dataset save result
        self.checkpoint_dir= './Model/checkpoints/'
        self.result_log_dir= '/Model/results/'
        self.tensorboard_dir= '/Model/tensorboard/'         
        
        
args = argus()

# print(args.save)      
# logger.info("Model Config: {}".format(args))
        
# create directory
for directory in [args.checkpoint_dir, args.result_log_dir, args.tensorboard_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(model, train_q_data, train_qa_data, 
            valid_q_data, valid_qa_data, result_log_path, args):
    saver = tf.compat.v1.train.Saver()
    best_loss = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0

    with open(result_log_path, 'w') as f:
        result_msg = "{},{},{},{},{},{},{}\n".format(
            'epoch', 
            'train_auc', 'train_accuracy', 'train_loss',
            'valid_auc', 'valid_accuracy', 'valid_loss'
        )
        f.write(result_msg)
    for epoch in range(args.n_epochs):
        
        # train_loss, train_accuracy, train_auc, train_ability, train_diff = run_model(
        #     model, args, train_q_data, train_qa_data, mode='train'
        # )
        # valid_loss, valid_accuracy, valid_auc, valid_ability, valid_diff = run_model(
        #     model, args, valid_q_data, valid_qa_data, mode='valid'
        # )

        train_loss, train_ability, train_diff = run_model(
            model, args, train_q_data, train_qa_data, mode='train'
        )
        valid_loss, valid_ability, valid_diff = run_model(
            model, args, valid_q_data, valid_qa_data, mode='valid'
        )


        train_auc = 0
        train_accuracy = 0
        valid_auc = 0
        valid_accuracy = 0

        # add to log
        msg = "\n[Epoch {}/{}] Training result:      AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch+1, args.n_epochs, train_auc*100, train_accuracy*100, train_loss
        )
        msg += "\n[Epoch {}/{}] Validation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch+1, args.n_epochs, valid_auc*100, valid_accuracy*100, valid_loss
        )
        logger.info(msg)

        # write epoch result
        with open(result_log_path, 'a') as f:
            result_msg = "{},{},{},{},{},{},{}\n".format(
                epoch, 
                train_auc, train_accuracy, train_loss,
                valid_auc, valid_accuracy, valid_loss
            )
            f.write(result_msg)

        # add to tensorboard
        tf_summary = tf.compat.v1.Summary(
            value=[
                tf.compat.v1.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.compat.v1.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.compat.v1.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                tf.compat.v1.Summary.Value(tag="valid_loss", simple_value=valid_loss),
                tf.compat.v1.Summary.Value(tag="valid_auc", simple_value=valid_auc),
                tf.compat.v1.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy),
            ]
        )
        model.tensorboard_writer.add_summary(tf_summary, epoch)
        
        # save the model if the loss is lower
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_accuracy
            best_auc = valid_auc
            best_epoch = epoch+1

            if args.save:
                model_dir = "ep{:03d}-auc{:.0f}-acc{:.0f}".format(
                    epoch+1, valid_auc*100, valid_accuracy*100,
                )
                model_name = "Deep-IRT"
                save_path = os.path.join(args.checkpoint_dir, model_dir, model_name)
                saver.save(sess=model.sess, save_path=save_path)

                logger.info("Model improved. Save model to {}".format(save_path))
            else:
                logger.info("Model improved.")

    # print out the final result
    msg = "Best result at epoch {}: AUC: {:.2f}\t Accuracy: {:.2f}\t Loss: {:.4f}\t Valid ability: {}\y Valid Difficulty: {}".format(
        best_epoch, best_auc*100, best_acc*100, best_loss, valid_ability, valid_diff
    )
    logger.info(msg)
    return best_auc, best_acc, best_loss, valid_ability, valid_diff

def cross_validation(num_of_data):
    tf.random.set_seed(1234)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    aucs, accs, losses = list(), list(), list()
    abilities, diffs = list(), list()
    for i in range(num_of_data):
        tf.compat.v1.reset_default_graph()
        logger.info("Cross Validation {}".format(i+1))
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i)+'.csv')

        with tf.compat.v1.Session(config=config) as sess:
            data_loader = DataLoader(args.n_questions, args.seq_len, ',')
            model = DeepIRTModel(args, sess, name="Deep-IRT")
            sess.run(tf.compat.v1.global_variables_initializer())
            if args.train:
                train_data_path = os.path.join(args.data_dir, args.data_name+'_train{}.csv'.format(i))
                valid_data_path = os.path.join(args.data_dir, args.data_name+'_valid{}.csv'.format(i))
                logger.info("Reading {} and {}".format(train_data_path, valid_data_path))

                train_q_data, train_qa_data = data_loader.load_data(train_data_path)
                valid_q_data, valid_qa_data = data_loader.load_data(valid_data_path)

                auc, acc, loss, ability, diff = train(
                    model, 
                    train_q_data, train_qa_data, 
                    valid_q_data, valid_qa_data, 
                    result_log_path=result_csv_path,
                    args=args
                )

                aucs.append(auc)
                accs.append(acc)
                losses.append(loss)
                abilities.append(ability)
                diffs.append(diff)
                
    cross_validation_msg = "Cross Validation Result:\n"
    cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs)*100, np.std(aucs)*100)
    cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs)*100, np.std(accs)*100)
    cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    logger.info(cross_validation_msg)

    # write result
    result_msg = datetime.datetime.now().strftime("%Y-%m-%dT%H%M") + ','
    result_msg += str(args.dataset) + ','
    result_msg += str(args.memory_size) + ','
    result_msg += str(args.key_memory_state_dim) + ','
    result_msg += str(args.value_memory_state_dim) + ','
    result_msg += str(args.summary_vector_output_dim) + ','
    result_msg += str(np.average(aucs)*100) + ','
    result_msg += str(np.std(aucs)*100) + ','
    result_msg += str(np.average(accs)*100) + ','
    result_msg += str(np.std(accs)*100) + ','
    result_msg += str(np.average(losses)) + ','
    result_msg += str(np.std(losses)) + '\n'
    with open('DeepIRT/Model/results/all_result.csv', 'a') as f:
        f.write(result_msg)
    
    return abilities, diffs

# if __name__=='__main__':
#    abilities = cross_validation()


# print(abilities)






