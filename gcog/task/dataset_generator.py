import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import gcog.task.config as config
import gcog.task.task_operators as task
import gcog.task.task_generator as taskgen
import networkx as nx
import random
import sys
import torch

class DatasetSampler(Dataset):
    """
    Class that encapsulates both task instance and its corresponding image sets (to save)

    Args:
        metatasks (list of task.MetaTask):   list of task.MetaTask to include in the training set (see task_operators.task for info)
        distractors (int):  number of distractors to include per trial (default: none, which randomly chooses between 1-10)
    """
    def __init__(self, 
                 metatasks, 
                 n_distractors=None):
        self.metatasks = metatasks
        self.n_distractors = 0 if n_distractors is None else n_distractors

    def __len__(self):
        return len(self.metatasks)

    def __getitem__(self, idx, distractor_range=False):
        """
        idx (int) : index of dataset
        distractor_range (bool): whether to randomly sample distractors with the range [0,5] with each sample
        """

        #### Get training items
        metatask = self.metatasks[idx]
        outputs = self.get_sample(metatask, idx, distractor_range=distractor_range)
        rule_array = outputs[0]
        stim_array = outputs[1]
        target_idx = outputs[2]
        return rule_array, stim_array, target_idx, idx#, df_metadata

    def get_sample(self, task, idx, distractor_range=False):
        if distractor_range:
            n_distractors = random.choice(range(self.n_distractors+1))
        else:
            n_distractors = self.n_distractors
        objset = task.generate_objset(n_distractor=n_distractors)
        target = task.get_target(objset)[0]
        rule_array =  self.convert_rule_input_idx(task).T

        # stim array -> 100 x colorshape x time
        stim_array = objset.objset2vecID().reshape(config.GRIDSIZE_X*config.GRIDSIZE_Y,-1,len(config.ALL_TIME))
        ####
        # Add EOS token, first zeropad stim_array
        stim_array_zeropad = np.zeros((stim_array.shape[0]+1,stim_array.shape[1]+1,stim_array.shape[2]))
        for t in range(len(config.ALL_TIME)):
            tmp = np.pad(stim_array[:,:,t],((0,1),(0,1)),'constant')
            stim_array_zeropad[:,:,t] = tmp
            stim_array_zeropad[-1,-1,t] = 1 # EOS token is the last embedding dim, and last token
        stim_array = stim_array_zeropad
        ####

        # flatten all but last 
        if isinstance(target,object) and not isinstance(target,bool): # probably shape or color attribute
            target_idx = config.OUTPUT_UNITS.index(target.value)
        else:
            target_idx = config.OUTPUT_UNITS.index(target)

        #path, num_computations, select_seq = self.traverse_path(task, objset)
        #num_obj = self.compute_num_objects(objset)

        #df_metadata = {}
        #df_metadata['task_idx'] = []
        #df_metadata['path'] = []
        #df_metadata['num_computations'] = []
        #df_metadata['select_seq'] = []
        #df_metadata['num_objects'] = []
        ##
        #df_metadata['task_idx'].append(idx)
        #df_metadata['path'].append(path)
        #df_metadata['num_computations'].append(num_computations)
        #df_metadata['select_seq'].append(select_seq)
        #df_metadata['num_objects'].append(num_obj)

        return rule_array, stim_array, target_idx

    def traverse_path(self,metatask,objset):
        """
        Function that takes in a specific task instance and 
        parses the path that's traversed for a given objset

        Args:
            metatask (taskgen.MetaTask)   :   an instance of the MetaTask class
            objset  (stim.ObjectSet)    :   an instance of the ObjectSet class
        
        Returns:
            string of traversed path
        """
        traversed_path = []
        select_sequence = []
        num_operations = []
        select_sequence = []
        current_operator = metatask._operator
        end_of_tree = False
        while not end_of_tree:
            if isinstance(current_operator,task.Switch):
                traversed_path.append(current_operator.child[0].__class__.__name__)
                num_operations.append(1)
                # traverse nonswitch node until select
                traversed_path, num_operations, select_sequence = self._traverse_nonswitch(current_operator.child[0], 
                                                                                          traversed_path, 
                                                                                          num_operations,
                                                                                          select_sequence)
                statement = current_operator.child[0](objset)
                if statement:
                    current_operator = current_operator.child[1] # if true, then left branch
                else:
                    current_operator = current_operator.child[2] # if false, then right branch
                traversed_path.append('IfElse')
                num_operations.append(1)
            else:
                traversed_path.append(current_operator.__class__.__name__)
                num_operations.append(1)
                # traverse nonswitch node until select
                traversed_path, num_operations, select_sequence = self._traverse_nonswitch(current_operator, 
                                                                                           traversed_path, 
                                                                                           num_operations,
                                                                                           select_sequence)
                end_of_tree = True
        return traversed_path, num_operations, select_sequence

    def compute_num_objects(self,objset):
        num_objects_per_window = []
        for when in config.ALL_TIME:
            num_objects = len(objset.dict[when])
            num_objects_per_window.append(num_objects)
        return num_objects_per_window

    def _traverse_nonswitch(self,current_operator, traversed_path, num_operations, select_sequence):
        num_parents = 1
        while not isinstance(current_operator, task.Select):
            num_child_operators = len(current_operator.child) * num_parents
            traversed_path.append(current_operator.child[0].__class__.__name__)
            num_operations.append(num_child_operators)
            num_parents = num_child_operators
            current_operator = current_operator.child[0]
        # keep track of what objects were selected
        if isinstance(current_operator, task.Select):
            select_str = []
            if current_operator.when: select_str.append(current_operator.when) 
            if current_operator.color.value: select_str.append(current_operator.color.value) 
            if current_operator.shape.value: select_str.append(current_operator.shape.value) 
            if current_operator.loc.value: select_str.append(str(current_operator.loc.value)) 
            select_sequence.append('_'.join(select_str))
        return traversed_path, num_operations, select_sequence

    def convert_rule_input_idx(self,metatask):
        """
        helper function to convert the networkX DAG into model input array
        this function formats the DAG into a topological generation (see nx.topological_generations)
        the output is organized as operators x topological generation, where operators
        refers to which operators are active in a particular part of the tree.

        Returns:
            task_matrix (numpy array): input_array * number of nodes in tree matrix
        """
        task_graph = metatask.DAG
        topological_sort = list(nx.topological_sort(task_graph))
        task_matrix = [] # entire input matrix
        for node in topological_sort:
            operator = metatask.DAG.nodes[node]['operator']
            node_id = metatask.DAG.nodes[node]['node_id']
            # fill operator element
            operator_idx = config.ALL_OPERATORS.index(node_id) + 1
            # fill select operators
            select_arrays = []
            for _ in range(config.MAX_SELECT_OPS): # create empty array
                select_arrays.append(np.zeros((config.SELECT_ARRAY_LENGTH,)))

            # Get operators are special case -- need to get select operators from operator's child (get op's child)
            if node_id in config.GETSHAPE + config.GETCOLOR:
                if node_id in ['isshape','iscolor','getcolor','getshape']:
                    select_arrays[0] = self._convert_select_to_idx(operator.objs)
                else:
                    for count, get in enumerate(operator.gets):
                        # gets only has one selector 
                        select_arrays[count] = self._convert_select_to_idx(get.objs)
            elif node_id in config.CONNECTOR_OPERATORS:
                pass # nothing to select
            else:
                if isinstance(operator.objs,list):
                    for count, o in enumerate(operator.objs): # fill in empty array for each select operator
                        select_arrays[count] = self._convert_select_to_idx(o)
                else:
                    select_arrays[0] = self._convert_select_to_idx(operator.objs)
            # flatten
            select_array = np.asarray(select_arrays).reshape(-1)
            # combine task operation + select operators
            input_array = np.hstack((operator_idx,select_array))
            # add to input matrix
            task_matrix.append(input_array)

        task_matrix = np.asarray(task_matrix).T # dimensions are one binary vectors X num nodes in tree

        ####
        # Add EOS token, first zeropad task_matrix
        task_matrix = np.pad(task_matrix,((0,1),(0,1)),'constant')
        task_matrix[-1,-1] = 1 # EOS token is the last embedding dim, and last token

        return task_matrix 

    def _convert_select_to_idx(self,obj):
        """Converts a select object to an input array"""
        select_array = np.zeros((config.SELECT_ARRAY_LENGTH,)) + 1
        if obj.shape.has_value:
            idx_shape = config.ALL_SHAPES.index(obj.shape.value) + 1
            select_array[0] = idx_shape
        if obj.color.has_value:
            idx_col = config.ALL_COLORS.index(obj.color.value) + 1
            select_array[1] = idx_col
        if obj.when is not None:
            idx_when = config.ALL_TIME.index(obj.when) + 1
            select_array[2] = idx_when
        return select_array
                
class CompTreeDataset(DatasetSampler):
    """
    Class to specify a compositional task tree structure. Primarily used for distractor generalization and productive comp. gen

    ntrials (int) : Number of trials to sample from *for each operator*. 
                    Thus, if there are 8 operators, the dataset will be of length 8 * ntrials (default ntrials=100)
                    Trials select a randomly sampled objects (e.g., red 'a') to be paired with an operator (e.g., 'exist')
    metatasks (list) : task list. If a task list was initialized before (and you would like to re-use it), place it here. Default is None, which will iterate through all possible operators
    location (bool) : If True, sample all objects (shape x color) from a uniformly distributed location grid
    tree_depth (int) : Creates a task tree of this length. Note that task trees must be of odd length (default: 1)
    distractor_range (int) : If True, all generated stimuli in this set will have a range of distractors of [1,n_distractors)
    n_distractors (int) : Number (or max number, if distractor_range=True) of distractors to be included in the image set
    nfeatures (int) : This limits the number of object feature dimension (color or shapes) to this int.
                    In other words, if nfeatures is 10, then only 10 colors/shapes will be sampled from (for faster task training)
                    Default is 0, which samples from all possible colors/shapes
    """
    def __init__(self, 
                 ntrials=100,
                 metatasks=None,
                 location=True,
                 tree_depth=3,
                 distractor_range=False,
                 n_distractors=None,
                 nfeatures=0):
        self.ntrials = ntrials
        self.location = location
        self.tree_depth = tree_depth
        self.nfeatures = nfeatures
        if metatasks is None:
            self._init_dataset()
        else:
            self.metatasks = metatasks
        super(CompTreeDataset,self).__init__(self.metatasks)
        self.n_distractors = 0 if n_distractors is None else n_distractors
        self.distractor_range = distractor_range

    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):
        return super().__getitem__(idx,distractor_range=self.distractor_range)

    def _init_dataset(self):
        """
        Create a list of all possible task metatasks from which to sample images from
        """
        metatasks = []
        tree_indices = []
        #### construct single node trees
        if self.tree_depth == 1:
            all_selectors = taskgen.generate_select_sets(config.ALL_TIME,randomize=True,location=self.location,
                                                         nfeatures=self.nfeatures)
            op_indices = []
            op_idx = 0
            for op_id in config.ALL_TASK_OPS:
                #for when in all_selectors:
                #    for obj in all_selectors[when]:
                #        selectors = {}
                #        selectors[when] = [obj] 
                for _ in range(self.ntrials):
                    selectors = {}
                    # randomly subsample selectors
                    for when in all_selectors:
                        selectors[when] = list(np.random.choice(all_selectors[when],1))
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                    ending_operators=None,
                                                    min_depth=1,
                                                    all_selectors=selectors,
                                                    whens=config.ALL_TIME,
                                                    num_selects_per_node=1)

                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                        tree_indices.append(self.tree_depth)
                op_idx += 1
            self.op_indices = op_indices

        #### construct multi node trees (if specified)
        if self.tree_depth > 1:
            for _ in range(self.ntrials*len(config.ALL_TASK_OPS)):
                all_selectors = taskgen.generate_select_sets_fortree(config.ALL_TIME,randomize=True,location=self.location,
                                                                     nfeatures=self.nfeatures)
                # randomly subsample selectors
                selectors = {}
                for when in all_selectors:
                    if self.tree_depth>9:
                        selectors[when] = list(np.random.choice(all_selectors[when],1000))
                    else:
                        selectors[when] = list(np.random.choice(all_selectors[when],100))
                #
                task_graph = None
                while task_graph is None:
                    # Sometimes it samples too many of the wrong operators, if so resample
                    try:
                        task_graph = taskgen.TaskGraph(starting_operators=config.STARTING_OPS,
                                                       ending_operators=config.ENDING_OPS,
                                                       min_depth=self.tree_depth,
                                                       all_selectors=selectors,
                                                       whens=config.ALL_TIME,
                                                       num_selects_per_node=None)
                        tree_indices.append(self.tree_depth)
                    except:
                        continue
                metatask = taskgen.MetaTask(task_graph)
                metatasks.append(metatask)

        self.tree_indices = tree_indices
        self.metatasks = metatasks

class CompTreeDatasetSubset(DatasetSampler):
    """
    Class to specify specific subsets of a compositional task tree structure. 
    Primarily used for systematic compositional generalization on trees of depth > 1

    ntrials (int) : Number of trials to sample from *for each operator*. 
                    Thus, if there are 8 operators, the dataset will be of length 8 * ntrials (default ntrials=100)
                    Trials select a randomly sampled objects (e.g., red 'a') to be paired with an operator (e.g., 'exist')
    metatasks (list) : task list. If a task list was initialized before (and you would like to re-use it), place it here. Default is None, which will iterate through all possible operators
    location (bool) : If True, sample all objects (shape x color) from a uniformly distributed location grid
    tree_depth (int) : Creates a task tree of this length. Note that task trees must be of odd length (default: 1)
    distractor_range (int) : If True, all generated stimuli in this set will have a range of distractors of [1,n_distractors)
    n_distractors (int) : Number (or max number, if distractor_range=True) of distractors to be included in the image set
    subset (int, 1 or 2): Which subset of task trees to sample from. 
                            If subset=1, then task trees include operators exist, sumeven, producteven, getcolor, getshape
                            If subset=2, then task trees include sumodd, productodd, exist, and go
    nfeatures (int) : This limits the number of object feature dimension (color or shapes) to this int.
                    In other words, if nfeatures is 10, then only 10 colors/shapes will be sampled from (for faster task training)
                    Default is 0, which samples from all possible colors/shapes
    """
    def __init__(self, 
                 ntrials=32,
                 metatasks=None,
                 location=True,
                 tree_depth=3,
                 distractor_range=False,
                 n_distractors=None,
                 subset=1,
                 nfeatures=0):
        self.ntrials = ntrials
        self.location = location
        self.tree_depth = tree_depth
        self.subset = subset
        self.nfeatures = nfeatures
        #### 
        self.STARTING_OPS1 = ['exist',
                             'sumeven', 'producteven'
                             ] 
        self.ENDING_OPS1 = [
                           'sumeven', 'producteven',
                           'getcolor', 'getshape'] 

        self.STARTING_OPS2 = ['sumodd','productodd'
                             ] 
        self.ENDING_OPS2 = ['exist',
                           'sumodd','productodd',
                           'go'] 
        if metatasks is None:
            self._init_dataset()
        else:
            self.metatasks = metatasks
        super(CompTreeDatasetSubset,self).__init__(self.metatasks)

        self.n_distractors = 0 if n_distractors is None else n_distractors
        self.distractor_range = distractor_range


    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):
        return super().__getitem__(idx,distractor_range=self.distractor_range)

    def _init_dataset(self):
        """
        Create a list of all possible task metatasks from which to sample images from
        """
        metatasks = []
        tree_indices = []
        #### construct single node trees
        if self.tree_depth == 1:
            all_selectors = taskgen.generate_select_sets(config.ALL_TIME,randomize=True,location=self.location,nfeatures=self.nfeatures)
            op_indices = []
            op_idx = 0
            for op_id in config.ALL_TASK_OPS:
                for _ in range(self.ntrials):
                    selectors = {}
                    # randomly subsample selectors
                    for when in all_selectors:
                        selectors[when] = list(np.random.choice(all_selectors[when],1))
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                    ending_operators=None,
                                                    min_depth=1,
                                                    all_selectors=selectors,
                                                    whens=config.ALL_TIME,
                                                    num_selects_per_node=1)

                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                        tree_indices.append(self.tree_depth)
                op_idx += 1
            self.op_indices = op_indices

        if self.tree_depth > 1:
            for _ in range(self.ntrials*len(config.ALL_TASK_OPS)):
                all_selectors = taskgen.generate_select_sets_fortree(config.ALL_TIME,randomize=True,location=self.location,nfeatures=self.nfeatures)
                # randomly subsample selectors
                selectors = {}
                for when in all_selectors:
                    if self.tree_depth>9:
                        selectors[when] = list(np.random.choice(all_selectors[when],1000))
                    else:
                        selectors[when] = list(np.random.choice(all_selectors[when],100))
                #
                task_graph = None
                while task_graph is None:
                    # Sometimes it samples too many of the wrong operators, if so resample
                    try:
                        # If subset is 1, make sure that certain starting operators are paired with ending operators
                        # This tests for systematicity
                        if self.subset==1:
                            if random.random()>0.5:
                                task_graph = taskgen.TaskGraph(starting_operators=self.STARTING_OPS1,
                                                            ending_operators=self.ENDING_OPS1,
                                                            min_depth=self.tree_depth,
                                                            all_selectors=selectors,
                                                            whens=config.ALL_TIME,
                                                            num_selects_per_node=None)
                                tree_indices.append(self.tree_depth)
                            else:
                                task_graph = taskgen.TaskGraph(starting_operators=self.STARTING_OPS2,
                                                            ending_operators=self.ENDING_OPS2,
                                                            min_depth=self.tree_depth,
                                                            all_selectors=selectors,
                                                            whens=config.ALL_TIME,
                                                            num_selects_per_node=None)
                                tree_indices.append(self.tree_depth)
                        # If subset is 2, make sure that certain starting operators are paired with ending operators
                        if self.subset==2:
                            if random.random()>0.5:
                                task_graph = taskgen.TaskGraph(starting_operators=self.STARTING_OPS1,
                                                            ending_operators=self.ENDING_OPS2,
                                                            min_depth=self.tree_depth,
                                                            all_selectors=selectors,
                                                            whens=config.ALL_TIME,
                                                            num_selects_per_node=None)
                                tree_indices.append(self.tree_depth)
                            else:
                                task_graph = taskgen.TaskGraph(starting_operators=self.STARTING_OPS2,
                                                            ending_operators=self.ENDING_OPS1,
                                                            min_depth=self.tree_depth,
                                                            all_selectors=selectors,
                                                            whens=config.ALL_TIME,
                                                            num_selects_per_node=None)
                                tree_indices.append(self.tree_depth)
                    except:
                        continue
                metatask = taskgen.MetaTask(task_graph)
                metatasks.append(metatask)

        self.tree_indices = tree_indices
        self.metatasks = metatasks


class OperatorSystematicity(DatasetSampler):
    """
    Class to measure the systematicity of individual operators
    Used for systematic compositional generalization on trees of depth = 1

    metatasks (list) : task list. If a task list was initialized before (and you would like to re-use it), place it here. Default is None, which will iterate through all possible operators
    location (bool) : If True, sample all objects (shape x color) from a uniformly distributed location grid
    distractor_range (int) : If True, all generated stimuli in this set will have a range of distractors of [1,n_distractors)
    n_distractors (int) : Number (or max number, if distractor_range=True) of distractors to be included in the image set
    subset (int, 1 or 2): Which subset of task operators to sample from. 
                          If subset=1, then task operators include exist, sumeven, producteven, getshape
                          If subset=2, then task operators include sumodd, productodd, go, getcolor
    nfeatures (int) : This limits the number of object feature dimension (color or shapes) to this int.
                    In other words, if nfeatures is 10, then only 10 colors/shapes will be sampled from (for faster task training)
                    Default is 0, which samples from all possible colors/shapes
    """
    def __init__(self, 
                 metatasks=None,
                 location=True,
                 distractor_range=False,
                 n_distractors=None,
                 subset=1,
                 nfeatures=0):
        self.location = location
        self.subset = subset
        self.nfeatures = nfeatures
        self.STARTING_OPS1 = ['exist','sumeven', 'producteven','getshape']
        self.STARTING_OPS2 = ['sumodd','productodd','go','getcolor']
        if metatasks is None:
            self._init_dataset()
        else:
            self.metatasks = metatasks
        super(OperatorSystematicity,self).__init__(self.metatasks)

        self.n_distractors = 0 if n_distractors is None else n_distractors
        self.distractor_range = distractor_range


    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):
        return super().__getitem__(idx,distractor_range=self.distractor_range)

    def _init_dataset(self):
        """
        Create a list of all possible task metatasks from which to sample images from
        """
        metatasks = []
        #### construct single node trees
        selectors1, selectors2 = self.generate_select_subsets(config.ALL_TIME,randomize=True,location=self.location,nfeatures=self.nfeatures)
        op_indices = []
        op_idx = 0
        ## SUBSET 1 -- uses starting_ops1 with selectors1 and starting_ops2 with selectors2
        if self.subset==1:
            for op_id in self.STARTING_OPS1:
                for when in selectors1:
                    selectors = {}
                    # iterate through every selector
                    for obj in selectors1[when]:
                        selectors[when] = [obj]
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                    ending_operators=None,
                                                    min_depth=1,
                                                    all_selectors=selectors,
                                                    whens=config.ALL_TIME,
                                                    num_selects_per_node=1)
                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                op_idx += 1
            for op_id in self.STARTING_OPS2:
                for when in selectors2:
                    selectors = {}
                    # iterate through every selector
                    for obj in selectors2[when]:
                        selectors[when] = [obj]
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                    ending_operators=None,
                                                    min_depth=1,
                                                    all_selectors=selectors,
                                                    whens=config.ALL_TIME,
                                                    num_selects_per_node=1)
                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                op_idx += 1

        ## SUBSET 2 -- uses starting_ops1 with selectors2 and starting_ops2 with selectors1
        elif self.subset==2:
            for op_id in self.STARTING_OPS1:
                for when in selectors2:
                    selectors = {}
                    # iterate through every selector
                    for obj in selectors2[when]:
                        selectors[when] = [obj]
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                       ending_operators=None,
                                                       min_depth=1,
                                                       all_selectors=selectors,
                                                       whens=config.ALL_TIME,
                                                       num_selects_per_node=1)
                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                op_idx += 1
            for op_id in self.STARTING_OPS2:
                for when in selectors1:
                    selectors = {}
                    # iterate through every selector
                    for obj in selectors1[when]:
                        selectors[when] = [obj]
                        #
                        task_graph = taskgen.TaskGraph(starting_operators=[op_id],
                                                    ending_operators=None,
                                                    min_depth=1,
                                                    all_selectors=selectors,
                                                    whens=config.ALL_TIME,
                                                    num_selects_per_node=1)
                        metatask = taskgen.MetaTask(task_graph)
                        metatasks.append(metatask)
                        op_indices.append(op_idx)
                op_idx += 1

        self.op_indices = op_indices
        self.metatasks = metatasks

    
    def generate_select_subsets(self, whens, randomize=True, location=False,nfeatures=0):
        """
        Generate two subsets of unique select instances to select specific objects during specific screen (time)
        
        Args:
            n_objects       :       Number of objects to produce (if None, produce all possible combos)
            whens           :       list of available times 
            randomize (bool):       if selectors should have randomized ordering
            nfeatures         :       Number of features to select from each dimension (color, shape)
        """
        all_colors = config.ALL_COLORS.copy()
        all_shapes = config.ALL_SHAPES.copy()

        if nfeatures>0:
            all_colors = all_colors[:nfeatures] 
            all_shapes = all_shapes[:nfeatures]
        
        n_subset_colors = round(len(all_colors)/2)
        n_subset_shapes = round(len(all_shapes)/2)

        colors_subset1 = all_colors[:n_subset_colors]
        colors_subset2 = all_colors[n_subset_colors:]
        shapes_subset1 = all_shapes[:n_subset_shapes]
        shapes_subset2 = all_shapes[n_subset_shapes:]

        # Create full list of locations
        if location:
            locations = []
            for x in range(config.GRIDSIZE_X):
                for y in range(config.GRIDSIZE_Y):
                    locations.append((y,x))

            if randomize:
                random.shuffle(locations)

        # Create two lists (subsets) of objects to sample from, which collectively includes all possible color/shape combos
        colorshapes1 = []
        #for shape in shapes_subset1:
        for shape in all_shapes:
            for color in colors_subset1:
                colorshapes1.append(tuple((color,shape)))

        colorshapes2 = []
        for shape in shapes_subset2:
            #for color in colors_subset2:
            for color in all_colors:
                colorshapes2.append(tuple((color,shape)))

        select_list1 = {}
        for when in set(whens):
            select_list1[when] = []
            colorshapes_ind = list(range(len(colorshapes1)))
            for i in range(len(colorshapes_ind)):
                if randomize:
                    tmp_colorshapes_ind = random.choice(colorshapes_ind) # randomly select colorshape index
                    colorshapes_ind.remove(tmp_colorshapes_ind)
                else:
                    tmp_colorshapes_ind = colorshapes_ind[i]
                #
                color = colorshapes1[tmp_colorshapes_ind][0]
                shape = colorshapes1[tmp_colorshapes_ind][1]
                #
                if location:
                    for loc in locations:
                        obj_params = {'color':color,'shape':shape, 'when':when, 'loc':loc}
                        select_list1[when].append(obj_params)
                else:
                    obj_params = {'color':color,'shape':shape, 'when':when}
                    select_list1[when].append(obj_params)

        select_list2 = {}
        for when in set(whens):
            select_list2[when] = []
            colorshapes_ind = list(range(len(colorshapes2)))
            for i in range(len(colorshapes_ind)):
                if randomize:
                    tmp_colorshapes_ind = random.choice(colorshapes_ind) # randomly select colorshape index
                    colorshapes_ind.remove(tmp_colorshapes_ind)
                else:
                    tmp_colorshapes_ind = colorshapes_ind[i]
                #
                color = colorshapes2[tmp_colorshapes_ind][0]
                shape = colorshapes2[tmp_colorshapes_ind][1]
                #
                if location:
                    for loc in locations:
                        obj_params = {'color':color,'shape':shape, 'when':when, 'loc':loc}
                        select_list2[when].append(obj_params)
                else:
                    obj_params = {'color':color,'shape':shape, 'when':when}
                    select_list2[when].append(obj_params)

        return select_list1, select_list2

class ConcatenatedDataset(DatasetSampler):
    "Class to concatenate multiple datasets"
    def __init__(self, 
                 datasetlist):
        metatasks = [] 
        tree_indices = []
        max_nodes = 0
        for ds in datasetlist:
            metatasks.extend(ds.metatasks)
            tree_indices.extend(ds.tree_indices)
            if len(ds.metatasks[0].DAG.nodes)>max_nodes:
                max_nodes = len(ds.metatasks[0].DAG.nodes) + 1 # include EOS token
        self.metatasks = metatasks
        self.tree_indices = tree_indices
        self.distractor_range = ds.distractor_range
        self.max_depth = np.max(self.tree_indices)
        self.max_nodes = max_nodes
        super(ConcatenatedDataset,self).__init__(self.metatasks)

    def __len__(self):
        return super().__len__()

    def __getitem__(self,idx):

        rule_array, stim_array, target_idx, idx = super().__getitem__(idx,distractor_range=self.distractor_range)

        # zero pad if it's a tree structure
        if hasattr(self,'max_depth'):
            if self.max_depth>1:
                tmp = torch.Tensor(rule_array).T # change to embedding x seq
                num_zero_pad = self.max_nodes - rule_array.shape[0] # pad by number of max nodes
                tmp_pad = torch.nn.functional.pad(tmp,(0,num_zero_pad),'constant',0)
                rule_array = tmp_pad.T

        #return super().__getitem__(idx,distractor_range=self.distractor_range)
        return rule_array, stim_array, target_idx, idx#, df_metadata


def collate_fn_rulepad(batch):
    '''
    For a given batch, this pads the rule sequence (batch[0]) of variable length so that the batch dimensions are the same
    '''
    rule_seq = []
    stim_seq = []
    target_seq = []
    op_indices = []
    for sample in batch:
        rule_seq.append(torch.Tensor(sample[0]))
        stim_seq.append(sample[1])
        target_seq.append(sample[2])
        op_indices.append(sample[3])
    new_rule_seq = torch.nn.utils.rnn.pad_sequence(rule_seq,batch_first=True)

    return new_rule_seq, torch.Tensor(stim_seq), torch.Tensor(target_seq).long(), np.asarray(op_indices)



