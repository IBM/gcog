import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import gcog.task.config as config
import gcog.task.task_generator as taskgen
import random
import gcog.models.train as train
import itertools
import gcog.models.analysis as analysis
from torch.utils.data import DataLoader, Subset
import time
import torch
import os

class Trainer(object):
    """
    class that iterates over tasks for each objects; 
    theoretically disentangles various object features 
    """
    def __init__(self, 
                 train_dataset,
                 train_loader,
                 train_evalset,
                 test_dataset,
                 testsets_all,
                 outputdir,
                 datasetname,
                 batch_size,
                 train_distractors=5,
                 trainsteps=0,
                 acc_cutoff=0.0,
                 n_epochs=0,
                 epoch0replay=True,
                 replay_multiplier=1,
                 device=torch.device('cuda'),
                 saveloop=True,
                 object_specific_loss=0.9,
                 op_specific_loss=0.7,
                 verbose=True):
        super(Trainer,self).__init__()
        #
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        #
        self.train_evalset = train_evalset
        self.test_dataset = test_dataset
        self.testsets_all = testsets_all
        # 
        self.outputdir = outputdir
        self.datasetname = datasetname
        #
        self.batch_size = batch_size
        self.train_distractors = train_distractors
        self.trainsteps = trainsteps
        self.acc_cutoff = acc_cutoff
        self.n_epochs = n_epochs
        self.epoch0replay = epoch0replay
        self.replay_multiplier = replay_multiplier
        self.device = device
        self.saveloop = saveloop
        self.object_specific_loss = object_specific_loss
        self.verbose = verbose
        #        
        self.min_accuracy = 0

        if acc_cutoff==0 and trainsteps==0:
            constraint_str = 'epochs'
            constraint_now = 0
            constraint = n_epochs
        if n_epochs==0 and trainsteps==0:
            constraint_str = 'cutoff'
            constraint_now = 0
            constraint = acc_cutoff
        if acc_cutoff==0 and n_epochs==0:
            constraint_str = 'trainsteps'
            constraint_now = 0
            constraint = trainsteps
        if n_epochs==0 and acc_cutoff==0 and trainsteps==0:
            raise Exception('trainsteps, n_epochs, and acc_cutoff cannot all equal 0; not possible')
        
        self.constraint_str = constraint_str
        self.constraint_now = constraint_now
        self.constraint = constraint

        # initiate task performance dict
        if not os.path.exists(outputdir+datasetname+'.csv'): 
            df_metadata = {}
            df_metadata['Epoch'] = []
            df_metadata['Training step'] = []
            df_metadata['Accuracy'] = []
            df_metadata['Condition'] = []
            df_metadata['Distractors'] = []
            df_metadata['Loss'] = []
            df_metadata['ElapsedTime'] = []
            self.epoch = 0
            self.trainstep_now = 0
        else:
            df_metadata = pd.read_csv(outputdir+datasetname+'.csv', index_col=[0])
            df_metadata = df_metadata.to_dict(orient='list')
            #
            self.epoch = df_metadata['Epoch'][-1] + 1
            self.trainstep_now = df_metadata['Training step'][-1]
            if constraint_str == 'epochs': 
                self.constraint_now = self.epoch
                self.constraint = self.epoch + n_epochs
        self.df_metadata = df_metadata
        
    def randomsplit_training(self, model, loss_constraint_step=None):
        """
        iterates over all objects
        """
        df_metadata = self.df_metadata
        while self.constraint_now < self.constraint:
            self.time0 = time.time()
            
            model, df_metadata = self.randomsplit_loop(model, df_metadata)

            self.epoch += 1
            if self.constraint_str=='epochs': self.constraint_now = self.epoch

            # Modify and update object specific loss
            if loss_constraint_step:
                if self.object_specific_loss+loss_constraint_step<=1:
                    self.object_specific_loss = self.object_specific_loss + loss_constraint_step

        state = {
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        torch.save(state,self.outputdir+self.datasetname+'.pt')
        df = pd.DataFrame(df_metadata)
        df.to_csv(self.outputdir+self.datasetname+'.csv')

    def randomsplit_loop(self, model, df_metadata):
        """
        A single epoch for the object/selector training
        """
        train_loader = self.train_loader
        #
        epoch = self.epoch
        self.time0 = time.time()
        loss_train = []
        #
        iterations_all = len(train_loader)
        iteration_count = 0
        for train_batch in train_loader:
            out = self.training_step(model,train_loader,
                                    df_metadata,
                                    loss_train,
                                    epoch,
                                    iteration_count,iterations_all,
                                    self.trainstep_now,
                                    train_batch=train_batch)
            model, train_loader, df_metadata, task_specific_loss_arr, self.trainstep_now = out
            iteration_count += 1

        return model, df_metadata

    def training_step(self, model, dataloader,
                      df_metadata,
                      loss_train,
                      epoch,
                      iteration_count,iterations_all,
                      trainstep_now,
                      train_batch=None,
                      verbose=True):
        train_evalset = self.train_evalset
        testsets_all = self.testsets_all
        train_distractors = self.train_distractors
        outputdir = self.outputdir 
        device = self.device
        min_accuracy = self.min_accuracy

        ## Select specific loader
        if train_batch is None:
            train_batch = next(dataloader)
        
        train_rules = train_batch[0].float().to(device)
        train_stim = train_batch[1].float().to(device)
        train_targets = train_batch[2].to(device)
        outputs, loss = train.train(model,
                                    train_rules,
                                    train_stim, 
                                    train_targets)    

        acc = analysis.accuracy_score(model,outputs,train_targets)
        #loss_train.append(acc.cpu().numpy())
        loss_train.append(loss)
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        trainstep_now += 1

        # record data and print training update
        if trainstep_now%501==0:  # for task rules
        #if trainstep_now%1==0:  # for selector updates
            ####
            # Evaluate training performance
            train_rules = train_evalset[0].float().to(device)
            train_stim = train_evalset[1].float().to(device)
            train_targets = train_evalset[2].float().to(device)
            train_indices = train_evalset[3]

            with torch.no_grad():
                outputs, hidden = model.forward(train_rules,
                                                train_stim)

            del hidden, train_rules, train_stim # free up mem

            # compute train accuracy
            acc = analysis.accuracy_score(model,outputs,train_targets,averaged=False)
            acc = acc.cpu().numpy()
            train_accuracy = np.mean(np.asarray(acc))*100.0
            # collect accuracy for this task + selector
            df_metadata['Epoch'].append(epoch)
            df_metadata['Training step'].append(trainstep_now)
            df_metadata['Accuracy'].append(train_accuracy)
            df_metadata['Condition'].append('train')
            df_metadata['Distractors'].append(train_distractors)
            df_metadata['Loss'].append(np.mean(loss_train))

            if verbose: 
                print(self.constraint_str, self.constraint_now, '/', self.constraint, '| Epoch', epoch, '|', self.datasetname,
                    '\n\tMean train accuracy:', np.mean(train_accuracy), '| loss:', np.mean(loss_train), '| grad norm:', total_norm,
                    '| lr:', model.optimizer.param_groups[0]["lr"],
                    '| num iteration:', iteration_count, '/', iterations_all)
                if hasattr(self.train_dataset,'op_indices'):
                    unique_op_indices = np.unique(self.train_dataset.op_indices)
                    trainset_op_indices = np.asarray(self.train_dataset.op_indices)[train_indices]
                    operator_str = ['\t\tOperator output']
                    #all_ops = config.SINGLE_OBJ_OPS+config.MULTI_OBJ_OPS
                    all_ops = config.ALL_TASK_OPS
                    for op_idx in unique_op_indices:
                        op_indices = np.where(trainset_op_indices==op_idx)[0]
                        tmp_acc = np.mean(acc[op_indices])
                        operator_str.append(' | ')
                        operator_str.append(all_ops[op_idx])
                        operator_str.append(': ' + str(round(tmp_acc,3)))
                    operator_str = ''.join(operator_str)
                    print(operator_str)
                if hasattr(self.train_dataset,'tree_indices'):
                    if len(np.unique(self.train_dataset.tree_indices))>1:
                        unique_tree_indices = np.unique(self.train_dataset.tree_indices)
                        trainset_tree_indices = np.asarray(self.train_dataset.tree_indices)[train_indices]
                        tree_str = ['\t\tTree output']
                        for tree_idx in unique_tree_indices:
                            tree_indices = np.where(trainset_tree_indices==tree_idx)[0]
                            tmp_acc = np.mean(acc[tree_indices])
                            tree_str.append(' | tree ')
                            tree_str.append(str(tree_idx))
                            tree_str.append(': ' + str(round(tmp_acc,3)))
                        tree_str = ''.join(tree_str)
                        print(tree_str)
            ####
            # Evaluate distractor generalization performance
            for n_dis in testsets_all:
                test_rules = testsets_all[n_dis][0].float().to(device)
                test_stim = testsets_all[n_dis][1].float().to(device)
                test_targets = testsets_all[n_dis][2].to(device)
                test_indices = testsets_all[n_dis][3]

                with torch.no_grad():
                    outputs, hidden = model.forward(test_rules,
                                                    test_stim)
                    del hidden, test_rules, test_stim # free up mem

                # compute test accuracy
                acc = analysis.accuracy_score(model,outputs,test_targets,averaged=False)
                acc = acc.cpu().numpy()
                test_accuracy = np.mean(np.asarray(acc))*100.0

                # store metadata
                df_metadata['Epoch'].append(epoch)
                df_metadata['Training step'].append(trainstep_now)
                df_metadata['Accuracy'].append(test_accuracy)
                df_metadata['Condition'].append('test')
                df_metadata['Distractors'].append(n_dis)
                df_metadata['Loss'].append(loss)

                if verbose:
                    if n_dis in testsets_all:
                        print('\tTest accuracy |', n_dis, 'distractors:', np.mean(test_accuracy))
                        if hasattr(self.test_dataset,'op_indices'):
                            unique_op_indices = np.unique(self.test_dataset.op_indices)
                            testset_op_indices = np.asarray(self.test_dataset.op_indices)[test_indices]
                            operator_str = ['\t\tOperator output']
                            #all_ops = config.SINGLE_OBJ_OPS+config.MULTI_OBJ_OPS
                            all_ops = config.ALL_TASK_OPS
                            for op_idx in unique_op_indices:
                                op_indices = np.where(testset_op_indices==op_idx)[0]
                                tmp_acc = np.mean(acc[op_indices])
                                operator_str.append(' | ')
                                operator_str.append(all_ops[op_idx])
                                operator_str.append(': ' + str(round(tmp_acc,3)))
                            operator_str = ''.join(operator_str)
                            print(operator_str)
                        if hasattr(self.test_dataset,'tree_indices'):
                            if len(np.unique(self.test_dataset.tree_indices))>1:
                                unique_tree_indices = np.unique(self.test_dataset.tree_indices)
                                testset_tree_indices = np.asarray(self.test_dataset.tree_indices)[test_indices]
                                tree_str = ['\t\tTree output']
                                for tree_idx in unique_tree_indices:
                                    tree_indices = np.where(testset_tree_indices==tree_idx)[0]
                                    tmp_acc = np.mean(acc[tree_indices])
                                    tree_str.append(' | tree ')
                                    tree_str.append(str(tree_idx))
                                    tree_str.append(': ' + str(round(tmp_acc,3)))
                                tree_str = ''.join(tree_str)
                                print(tree_str)

            
            timeend = time.time()
            if verbose:
                print('\tElapsed time (in epoch):', timeend-self.time0)

            df_metadata['ElapsedTime'].append(timeend-self.time0)
            for n_dis in testsets_all: df_metadata['ElapsedTime'].append(timeend-self.time0)

            if self.constraint_str=='cutoff': self.constraint_now = min_accuracy
            if self.constraint_str=='trainsteps': self.constraint_now = trainstep_now

            # Intermittent saving
            if self.saveloop:
                state = {
                    'state_dict': model.state_dict(),
                    'optimizer': model.optimizer.state_dict()
                }
                torch.save(state,self.outputdir+self.datasetname+'.pt')
                df = pd.DataFrame(df_metadata)
                df.to_csv(outputdir+self.datasetname+'.csv')
                model.train()
            
        
        return model, dataloader, df_metadata, loss_train, trainstep_now
                
        

