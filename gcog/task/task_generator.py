import numpy as np
import gcog.task.task_operators as task
import gcog.task.task_generator as taskgen
import gcog.task.dataset_generator as datagen
import gcog.task.stim_generator as stim
import gcog.task.config as config
from gcog.task.task_operators import Task
from collections import defaultdict
from collections import Counter
import warnings
import random
import networkx as nx
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout
warnings.filterwarnings("ignore") # runtime warning from graphviz


NODE_OPERATORS = [task.Exist, task.ExistAnd, task.ExistOr, task.ExistXor, 
                  task.SameColor, task.SameShape, task.NotSameColor, task.NotSameShape,
                  task.Go, 
                  task.SumEven, task.ProductEven, task.SumOdd, task.ProductOdd,
                  task.GetShape, task.GetColor, task.IsShape, task.IsColor] # task.AddEqual, task.SubtractEqual, task.MultiplyEqual

class TaskGraph(object):
    """
    Class that generates a task graph using a simple context-free grammar

        Args:
            starting_operators (list):   a list of operators (str) to be used in the task graph
            ending_operators (list):   a list of operators (str) to be used in the task graph
            connector_operators (list):   a list of operators (str) to be used in the task graph
            all selectors (list or None):   a list of dictionaries containing select attributes
            min_depth (int):   minimum depth of task graph
            num_selects_per_node (int or None):  *RARELY USED* specify the number of select operators to have per node. 
                                         Typically used only for one-rule pretraining.
                                         only specify if you know what you're doing.
    """
    def __init__(self,
                 starting_operators=config.STARTING_OPS,
                 ending_operators=config.ENDING_OPS, 
                 connector_operators=config.CONNECTOR_OPERATORS,
                 all_selectors=None,
                 whens=None,
                 min_depth=1,
                 num_selects_per_node=None):
        self.starting_operators = starting_operators
        self.ending_operators = ending_operators
        self.connector_operators = connector_operators
        self.min_depth = min_depth
        self.G = nx.DiGraph() # initialize DAG
        if all_selectors is None:
            self.all_selectors = generate_select_sets(whens=config.ALL_TIME) # store unique objects to sample from
        else:
            self.all_selectors = all_selectors
        self.num_selects_per_node = num_selects_per_node

        # create a list of available attributes to choose from
        self.whens = list(self.all_selectors.keys()) if whens is None else whens
       
        # For operators to 'get'
        available_colors = {}
        available_shapes = {}
        # For distractors to sample from
        available_distractors_colors = {}
        available_distractors_shapes = {}
        for when in self.whens:
            available_colors[when] = config.ALL_COLORS.copy()
            available_shapes[when] = config.ALL_SHAPES.copy()
            available_distractors_colors[when] = config.ALL_COLORS.copy()
            available_distractors_shapes[when] = config.ALL_SHAPES.copy()

        self.available_colors = available_colors
        self.available_shapes = available_shapes
        self.available_distractors_colors = available_distractors_colors
        self.available_distractors_shapes = available_distractors_shapes

        self.taken_locations = []

        self.construct_graph()

        self.verify_tree_uniqueness()

    def construct_graph(self):
        """
        main method to construct network x graph
        TODO -- this probably can be cleaned
        """
        starting_operators = self.starting_operators
        ending_operators = self.ending_operators
        connector_operators = self.connector_operators
        min_depth = self.min_depth

        #### Start construction
        # Add root node
        root_node = random.choice(starting_operators)
        node_counter = 0
        obj_counter = 0
        node_parent = 0  # special case
        node_counter, obj_counter = self.__add_node_to_graph(root_node, node_counter, obj_counter, node_parent=None)
        # root node special case
        node_counter += 1


        # now fill out one branch with min_depth
        path_length = len(nx.dag_longest_path(self.G))
        while path_length<min_depth:
            ### Now include a connector (i.e., switch)
            node = random.choice(self.connector_operators)
            node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
            node_parent = node_counter
            path_length += 1

            if path_length>=min_depth-1:
                # if G is long enough, then find ending operators (need two after a connector)
                node = random.choice(self.ending_operators)
                node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)

                # now repeat this for other node
                node = random.choice(self.ending_operators)
                node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                node_parent = node_counter
                path_length += 1
            else:
                # if G isn't long enough, then add additional operators
                node = random.choice(self.starting_operators)
                node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)

                # now repeat this for other node
                node = random.choice(self.starting_operators)
                node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                node_parent = node_counter
                path_length += 1

        # Above code block only identified one path of at least min_depth.
        # Now fill out the remaining task tree to guarantee all paths have at least min_depth
        sink_nodes = [node for node, outdegree in self.G.out_degree(self.G.nodes()) if outdegree==0]
        # remove any sink_nodes with longer paths than min_depth
        for sink in sink_nodes:
            path = list(nx.shortest_simple_paths(self.G, 0, sink))[0]
            if len(path)>=min_depth:
                sink_nodes.remove(sink)

        while sink_nodes:
            sink = sink_nodes[0]
            # in our tree-like DAG, we only have 1 unique path from source to sink
            path = list(nx.shortest_simple_paths(self.G, 0, sink))[0]
            path_length = len(path)
            if path_length<min_depth:
                node_parent = sink
                
                node = random.choice(self.connector_operators)
                node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                node_parent = node_counter
                path_length += 1

                if path_length>=min_depth-1:
                    # if G is long enough, then find ending operators (need two after a connector)
                    node = random.choice(self.ending_operators)
                    node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)

                    # now repeat this for other node
                    node = random.choice(self.ending_operators)
                    node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                    node_parent = node_counter

                    ## Reached limit, remove sink
                    sink_nodes.remove(sink)
                else:
                    # if G isn't long enough, then add additional operators
                    node = random.choice(self.starting_operators)
                    node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                    sink_nodes.append(node_counter)

                    # now repeat this for other node
                    node = random.choice(self.starting_operators)
                    node_counter, obj_counter = self.__add_node_to_graph(node, node_counter, obj_counter, node_parent)
                    node_parent = node_counter
                    
                    sink_nodes.append(node_counter)
                    sink_nodes.remove(sink)
            else:
                sink_nodes.remove(sink)

        self._check_graph_validity()
        self.__identify_available_distractors()

    def compute_average_path_length(self):
        """
        compute the average task graph depth (across all paths)
        """

        sink_nodes = [node for node, outdegree in self.G.out_degree(self.G.nodes()) if outdegree==0]
        # remove any sink_nodes with longer paths than min_depth
        paths = []
        for sink in sink_nodes:
            paths.append(list(nx.shortest_simple_paths(self.G, 0, sink))[0])


        nodes = self.G.nodes()
        path_lengths = []
        for path in paths:
            length = 0
            for node in path:
                operator = self.G.nodes[node]['operator'].op
                try:
                    length += operator.operator_length()
                except:
                    if issubclass(operator, task.Switch): 
                        length += 1
                        
            path_lengths.append(length)
        return np.mean(path_lengths)

    def __add_node_to_graph(self, node, node_counter, obj_counter, node_parent=None, num_select_objs=1):
        """
        private method that adds a node to the graph
        Args:
            node        :   node/operator (string)
            node_counter:   node id in networkx G   
            node_parent :   Parent of the node (to draw edge); None if root
            obj_counter (int):   object index   
            num_select_objs (int):  Number of objects to select (only possible for nodes that have more than 1 parent) (default=1)
        Returns:    
            node_counter
            obj_counter
        """
        if node not in config.MULTI_OBJ_OPS:
            if num_select_objs>1:
                raise Exception("This node cannot select more than one object")

        if node in config.MULTI_OBJ_OPS:
            # then set default to two objects
            num_select_objs = 2
        else:
            num_select_objs = 1

        if node in config.MULTI_OBJ_OPS and node in config.SINGLE_OBJ_OPS:
            # for example: add, subtract, multiply
            if self.num_selects_per_node is None:
                num_select_objs = random.choice([1,2])
            else:
                num_select_objs = self.num_selects_per_node

        # special case
        if node_counter!=0:
            node_counter += 1
            self.G.add_edge(node_parent, node_counter)

        try:
            available_whens = list(self.all_selectors.keys())
            when = np.random.choice(available_whens,1,p=config.ALL_TIME_PROB)[0]
        except:
            # This is for unit testing purposes only!
            #raise NotImplementedError("Not implemented for fewer or greater than 3 time slots",
            #                          "Need to adjust the constant values in stim_generator.py")
            available_whens = list(self.all_selectors.keys())
            when = np.random.choice(available_whens,1)[0]
        if node in self.connector_operators:
            self.G.add_node(node_counter, 
                            node_id=node,
                            selector_ids=[None])
        else:
            if num_select_objs==2:
                obj1 = self.all_selectors[when][obj_counter]
                if 'loc' in obj1:
                    loc1 = obj1['loc']
                    # make sure this is an available location
                    if loc1 in self.taken_locations:
                        loc1 = obj1['loc']
                        if len(self.taken_locations)==len(config.ALL_LOCATIONS):
                            raise Exception('ALL LOCATIONS TAKEN')
                        diffloc = False
                        while diffloc is False:
                            loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                            if loc1!=loc2 and loc2 not in self.taken_locations:
                                obj1['loc'] = loc2
                                diffloc = True
                selectors = {}
                selectors[when] = []
                selectors[when].append(obj1)
                # get 2nd selector
                obj2 = pick_2ndtask_selector(node,obj1,taken_locations=self.taken_locations)
                selectors[when].append(obj2)
                self.G.add_node(node_counter, 
                                node_id=node,
                                selector_ids=selectors[when][:num_select_objs])
                if 'loc' in obj1:
                    self.taken_locations.append(obj1['loc'])
                if 'loc' in obj2:
                    self.taken_locations.append(obj2['loc'])
                obj_counter += num_select_objs
            elif num_select_objs==1:
                obj1 = self.all_selectors[when][obj_counter]
                if 'loc' in obj1:
                    if obj1['loc'] in self.taken_locations:
                        loc1 = obj1['loc']
                        if len(self.taken_locations)==len(config.ALL_LOCATIONS):
                            raise Exception('ALL LOCATIONS TAKEN')
                        diffloc = False
                        while diffloc is False:
                            loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                            if loc1!=loc2 and loc2 not in self.taken_locations:
                                obj1['loc'] = loc2
                                diffloc = True
                        self.all_selectors[when][obj_counter]['loc'] = obj1['loc']
 

                self.G.add_node(node_counter, 
                                node_id=node,
                                selector_ids=self.all_selectors[when][obj_counter:obj_counter+num_select_objs])
                if 'loc' in self.all_selectors[when][obj_counter]: 
                    self.taken_locations.append(self.all_selectors[when][obj_counter]['loc'])
                obj_counter += num_select_objs
            else:
                raise Exception("Not Implemented for more than 2 selectors per operator")

        return node_counter, obj_counter

    def _check_graph_validity(self):
        """
        private method that searches nodes to make sure that get functions do not conflict with other
        existing select methods
        E.g., if Get(color='red') exists, this function will make sure that there is 
        no other Select(color='red') that exists
        """
        nodes = self.G.nodes

        get_attributes = self.__collect_get_attributes()

        for when in self.whens:
            self.available_colors[when] = list(set(self.available_colors[when])^set(get_attributes['color'][when]))
            self.available_shapes[when] = list(set(self.available_shapes[when])^set(get_attributes['shape'][when]))

        get_attributes = self.__replace_get_attributes(get_attributes)

        self.__replace_select_attributes(get_attributes)

    def __collect_get_attributes(self): 
        """collects all get attributes in the task tree"""
        nodes = self.G.nodes
        get_attributes = {}
        unique_times = list(np.unique(config.ALL_TIME))
        get_attributes['color'] = {} # need to identify when color/shape are present
        get_attributes['shape'] = {}
        for when in unique_times: get_attributes['color'][when] = []
        for when in unique_times: get_attributes['shape'][when] = []

        # first find all attributes that need to be retrieved in task graph
        for node in nodes:
            node_instance = nodes[node]
            node_str = node_instance['node_id']
            if node_str in config.GETCOLOR:
                for o in node_instance['selector_ids']:
                    # check if getting shape or color
                    attr_type = 'color'
                    when_obj = o['when']
                    attr = o['color']
                    get_attributes[attr_type][when_obj].append(attr)

            if node_str in config.GETSHAPE:
                for o in node_instance['selector_ids']:
                    # check if getting shape or color
                    attr_type = 'shape'
                    when_obj = o['when']
                    attr = o['shape']
                    get_attributes[attr_type][when_obj].append(attr)

        return get_attributes

    def __replace_get_attributes(self, get_attributes):
        """replaces any conflicting get attributes"""
        nodes = self.G.nodes

        # Now make sure there are no conflicts between get family attributes
        for node in nodes:
            node_instance = nodes[node]
            node_str = node_instance['node_id']
            for o in node_instance['selector_ids']:
                if node_str not in config.GETCOLOR + config.GETSHAPE: continue
                if node_str in config.GETCOLOR: attr_type = 'shape'
                if node_str in config.GETSHAPE: attr_type = 'color'
                when = o['when']
                attr = o[attr_type]
                counter_dict = Counter(get_attributes[attr_type][when])
                if counter_dict[attr]>1:
                    if attr_type == 'color':
                        if len(self.available_colors[when])==0:
                            raise Exception('Error with this task. Too many Get operators included',
                                             'The number of unique task attributes has run out.',
                                             'Please re-run or reduce the complexity of this task')
                        new_attr_str = random.choice(self.available_colors[when])
                        self.available_colors[when].remove(new_attr_str)
                    elif attr_type == 'shape':
                        if len(self.available_shapes[when])==0:
                            raise Exception('Error with this task. Too many Get operators included',
                                             'The number of unique task attributes has run out.',
                                             'Please re-run or reduce the complexity of this task')
                        new_attr_str = random.choice(self.available_shapes[when])
                        self.available_shapes[when].remove(new_attr_str)

                    get_attributes[attr_type][when].append(new_attr_str)
                    o[attr_type] = new_attr_str


            for o in node_instance['selector_ids']:
                if node_str not in config.GETCOLOR + config.GETSHAPE: continue
                if node_str in config.GETCOLOR: attr_type = 'shape'
                if node_str in config.GETSHAPE: attr_type = 'color'
                when = o['when']
                attr = o[attr_type]
                counter_dict = Counter(get_attributes[attr_type][when])
                if counter_dict[attr]>1:
                    if attr_type == 'color':
                        if len(self.available_colors[when])==0:
                            raise Exception('Error with this task. Too many Get operators included',
                                             'The number of unique task attributes has run out.',
                                             'Please re-run or reduce the complexity of this task')
                        new_attr_str = random.choice(self.available_colors[when])
                        self.available_colors[when].remove(new_attr_str)
                        new_attr = stim.Color(new_attr_str)
                    elif attr_type == 'shape':
                        if len(self.available_shapes[when])==0:
                            raise Exception('Error with this task. Too many Get operators included',
                                             'The number of unique task attributes has run out.',
                                             'Please re-run or reduce the complexity of this task')
                        new_attr_str = random.choice(self.available_shapes[when])
                        self.available_shapes[when].remove(new_attr_str)

                    get_attributes[attr_type][when].append(new_attr_str)
                    o[attr_type] = new_attr_str

        return get_attributes

    def __replace_select_attributes(self, get_attributes):
        """replaces all other conflicting select attributes"""
        nodes = self.G.nodes

        # Now iterate through graph again and make sure no other conflicts (would make task ill-posed)
        for node in nodes:
            node_instance = nodes[node]
            node_str = node_instance['node_id']
            # only care about select operators
            if node_str in config.GETCOLOR or config.GETSHAPE: 
                continue
            if node_str in self.connector_operators: # connector operators have no select
                continue
            
            select_objs = node_instance['selector_ids']
            for o in select_objs:
                when = o['when']
                if o['color'] in get_attributes['color'][when]:
                    #print('found color conflict, object', o, 'with get', o.when, get_attributes['color'][o.when])
                    if len(self.available_colors[when])==0:
                        raise Exception('Error with this task. Too many Get operators included',
                                         'The number of unique task attributes has run out.',
                                         'Please re-run or reduce the complexity of this task')
                    o['color'] = random.choice(self.available_colors[when]) # change color
                    self.available_colors[when].remove(o['color'])
                
                if o['shape'] in get_attributes['shape'][when]:
                    #print('found shape conflict, object', o, 'with get', o.when, get_attributes['shape'][o.when])
                    if len(self.available_shapes[when])==0:
                        raise Exception('Error with this task. Too many Get operators included',
                                         'The number of unique task attributes has run out.',
                                         'Please re-run or reduce the complexity of this task')
                    o['shape'] = random.choice(self.available_shapes[when]) # change shape
                    self.available_shapes[when].remove(o['shape'])

    def __identify_available_distractors(self):
        """Identifies available distractors that do not conflict with any get operators"""
        nodes = self.G.nodes
            
        for node in nodes:
            node_instance = nodes[node]
            node_str = node_instance['node_id']
            if node_str in config.GETCOLOR:
                select_objs = nodes[node]['selector_ids']
                for o in select_objs:
                    when = o['when']
                    # it's the attribute that's not being retrieved
                    unavailable_distractor = o['shape']
                    if unavailable_distractor in self.available_distractors_shapes[when]:
                        self.available_distractors_shapes[when].remove(unavailable_distractor)

            if node_str in config.GETSHAPE:
                select_objs = nodes[node]['selector_ids']
                for o in select_objs:
                    when = o['when']
                    # it's the attribute that's not being retrieved
                    unavailable_distractor = o['color']
                    if unavailable_distractor in self.available_distractors_colors[when]:
                        self.available_distractors_colors[when].remove(unavailable_distractor)

    def verify_tree_uniqueness(self):
        """
        Verify that the tree has no repeating operators.
        While task operators can be repeated, no operators x select operator (interactions) can be repeated
        """
        for i in self.G.nodes:
            for j in self.G.nodes:
                if i==j: continue
                # switch operators can be repeated
                if self.G.nodes[i]['node_id']=='switch': continue
                # make sure no operators are identical
                #if self.G.nodes[i]['node_id'] == self.G.nodes[j]['node_id']:
                #    raise Exception('This DAG has repeating operators. Not a problem, but makes task uninteresting')

    def plot_graph(self,
                   with_labels=True, 
                   node_color="white", 
                   edgecolors="white",
                   node_shape='s',
                   node_size=1000,
                   font_family="Arial",
                   font_size=10):
        """
        **kwargs  -- arguments for Network X draw input
        """
        pos = graphviz_layout(self.G, prog='dot')
        nodes = self.G.nodes
        labels = {}
        for node in nodes:
            tag = nodes[node]['node_id']
            nodeid = nodes[node]['node_id']
            if nodes[node]['selector_ids'][0] is not None:
                color = nodes[node]['selector_ids'][0]['color']
                shape = nodes[node]['selector_ids'][0]['shape']
                when = nodes[node]['selector_ids'][0]['when']
            if tag in ['go', 'exist','getshape','getcolor','isshape','iscolor']:
                string = [tag]
                string += ['\n']
                string += [when]
                if tag in ['isshape']:
                    string += [color]
                    string += ["is '"+ shape+"'"]
                elif tag in ['iscolor']:
                    string += ["'" + shape + "'"]
                    string += ["is "+ color]
                elif tag in ['getshape']:
                    string += [color]
                    string += ['object']
                elif tag in ['getcolor']:
                    string += [shape]
                else:
                    string += [color]
                    string += ['"'+shape+'"']
                string += ['?'] 
                string = ' '.join(string)
                labels[node] = string
            if tag in ['existand', 'existor','existxor']:
                if tag in  ['existand']:
                    conj = 'and'
                elif tag == 'existor':
                    conj = 'or'
                elif tag == 'existxor':
                    conj = 'xor'
                tag = 'exist'
                for o in range(len(nodes[node]['selector_ids'])-1):
                    color = nodes[node]['selector_ids'][o]['color']
                    shape = nodes[node]['selector_ids'][o]['shape']
                    when = nodes[node]['selector_ids'][o]['when']
                    tag += '\n' + when + ' ' + color + ' ' + shape + ' ' + conj
                    
                color = nodes[node]['selector_ids'][o+1]['color']
                shape = nodes[node]['selector_ids'][o+1]['shape']
                when = nodes[node]['selector_ids'][o+1]['when']
                tag += '\n' + when + ' ' + color + ' ' + shape
                labels[node] = tag 
            if tag in ['sameshape','samecolor','notsameshape','notsamecolor']:
                conj = 'and'
                for o in range(len(nodes[node]['selector_ids'])-1):
                    if nodeid in config.GETCOLOR:
                        shape = nodes[node]['selector_ids'][o]['shape']
                        color = ''
                    if nodeid in config.GETSHAPE:
                        color = nodes[node]['selector_ids'][o]['color']
                        shape = '' 
                    when = nodes[node]['selector_ids'][o]['when']
                    tag += '\n' + when + ' ' + color + ' ' + shape + ' ' + conj

                if nodeid in config.GETCOLOR:
                    shape = nodes[node]['selector_ids'][o+1]['shape']
                    color = ''
                if nodeid in config.GETSHAPE:
                    color = nodes[node]['selector_ids'][o+1]['color']
                    shape = '' 
                when = nodes[node]['selector_ids'][o+1]['when']
                tag += '\n' + when + ' ' + color + ' ' + shape
                labels[node] = tag 
            if tag == 'switch':
                labels[node] = 'if-then'
            if tag in ['add', 'subtract', 'multiply', 'divide']:
                raise NotImplemented("Stopped implementing this function")

            if tag in ['sumeven','producteven','sumodd','productodd']:
                if tag in  ['sumeven','sumodd']:
                    conj = '+'
                elif tag in ['producteven','productodd']:
                    conj = '*'
                
                if len(nodes[node]['selector_ids'])==1:
                    color = nodes[node]['selector_ids'][0]['color']
                    shape = nodes[node]['selector_ids'][0]['shape']
                    when = nodes[node]['selector_ids'][0]['when']
                    tag += '\n' + when + ' ' + color + ' ' + shape
                    labels[node] = tag 
                else:
                    tag = tag 
                    for o in range(len(nodes[node]['selector_ids'])-1):
                        color = nodes[node]['selector_ids'][o]['color']
                        shape = nodes[node]['selector_ids'][o]['shape']
                        when = nodes[node]['selector_ids'][o]['when']
                        tag += '\n' + when + ' ' + color + ' ' + shape + ' ' + conj
                    
                    color = nodes[node]['selector_ids'][o+1]['color']
                    shape = nodes[node]['selector_ids'][o+1]['shape']
                    when = nodes[node]['selector_ids'][o+1]['when']
                    tag += '\n' + when + ' ' + color + ' ' + shape
                    labels[node] = tag 
        nx.draw(self.G, pos=pos,labels=labels,with_labels=with_labels, 
                node_color=node_color, edgecolors=edgecolors, node_shape=node_shape, node_size=node_size,
                font_family=font_family,font_size=font_size)

class MetaTask(task.Task):
    """
    Creates a metatask of gcog.task_operators 
    Takes in a task_graph (the class above)
    based on a networkX DAG description (in this case a tree)

        Args:
            DAG    :   A directed acyclic graph (DAG) that describes a graph of task operations
    """

    def __init__(self,task_graph): 
        super(MetaTask,self).__init__()
        self.task_graph = deepcopy(task_graph)
        self.DAG = deepcopy(task_graph.G)
        self.whens = task_graph.whens 
        self.available_colors = task_graph.available_colors
        self.available_shapes = task_graph.available_shapes
        self.available_distractors_colors = task_graph.available_distractors_colors
        self.available_distractors_shapes = task_graph.available_distractors_shapes

        # Now fill in DAG with class operators
        for node in self.DAG.nodes:
            operator = self.return_operator(self.DAG.nodes[node])
            self.DAG.nodes[node]['operator'] = operator


        # First find a leaf of the tree
        reverse_sort = list(reversed(list(nx.topological_sort(self.DAG))))
        i = reverse_sort[0]
        # Set operator at the bottom of DAG
        operator = self.DAG.nodes[i]['operator']
        # Now via reverse topological order, update operators
        visited = []
        # dictionary to store subgraphs that don't reach the root node
        subgraphs = defaultdict(lambda: None)
        for i in reverse_sort:
            if i in visited:
                continue
            
            operator = self.DAG.nodes[i]['operator']
            visited.append(i)
            
            # Specify all types first (rather than instances)

            operator, subgraphs, visited = self.__generate_subgraph(operator, i, subgraphs, visited)

        self._operator = operator

    def __generate_subgraph(self, operator, i, subgraphs, visited):
        """
        helper function to traverse subgraphs from reverse topological sort

        -- this should probably be cleaned in future versions
        """
        if type(operator)==type:
            if issubclass(operator, task.Switch):
                switch_parent = list(self.DAG.predecessors(i)) # only 1 parent for switch
                switch_statement = self.DAG.nodes[switch_parent[0]]['operator']
                # find switch children
                switch_children = list(self.DAG.successors(i)) # 2 children for switch
                if type(switch_statement)==type:
                    switch_statement, subgraphs, visited = self.__generate_subgraph(switch_statement, switch_parent[0], subgraphs, visited)
                    # Need to alert the parents of the switch parents (i.e., switch_statement) that operator is now included in switch statement
                    parent_parent_switch_statement = list(self.DAG.predecessors(switch_parent[0]))
                    for parent in parent_parent_switch_statement: 
                        subgraphs[parent] = operator(switch_statement, subgraphs[switch_children[0]], subgraphs[switch_children[1]])
                        visited.append(parent)
                    visited.append(switch_parent[0])
                    # also notify current node of subgraph
                    subgraphs[i] = operator(switch_statement, subgraphs[switch_children[0]], subgraphs[switch_children[1]])
                    subgraphs[switch_parent[0]] = operator(switch_statement, subgraphs[switch_children[0]], subgraphs[switch_children[1]])
                    operator = operator(switch_statement, subgraphs[switch_children[0]], subgraphs[switch_children[1]])
                else:
                    visited.append(switch_parent[0])
                    #
                    operator = operator(switch_statement, subgraphs[switch_children[0]], subgraphs[switch_children[1]])
                    subgraphs[i] = operator
                    subgraphs[switch_parent[0]] = operator

        else:
            if isinstance(operator, tuple(NODE_OPERATORS)):
                children = list(self.DAG.successors(i)) # would be one child only
                if children:
                    if len(children)>1:
                        raise Exception('did not expect more than one child for these operators')
                    else:
                        operator = operator(subgraphs[children[0]])
                subgraphs[i] = operator

        return operator, subgraphs, visited

    def return_operator(self,node):
        """Identify and return operator for a specified dict (or NetworkX node object)"""
        operator_id = node['node_id']
        objs = []
        if operator_id in config.CONNECTOR_OPERATORS:
            pass
        else:
            for o in node['selector_ids']:
                color = stim.Color(o['color'])
                shape = stim.Shape(o['shape'])
                when = o['when']
                if 'loc' in o:
                    loc = stim.Loc(o['loc'])
                else:
                    loc = None
                obj = task.Select(color=color, shape=shape, when=when, loc=loc)
                objs.append(obj)

        if len(objs)==1:
            objs1 = objs[0]

        # specify a class instance
        if operator_id == 'go':
            op = task.Go(objs1)
        if operator_id == 'exist':
            op = task.Exist(objs1)
        if operator_id == 'existand':
            op = task.ExistAnd(objs) 
        if operator_id == 'existor':
            op = task.ExistOr(objs)
        if operator_id == 'existxor':
            op = task.ExistXor(objs)
        if operator_id == 'add':
            op = task.AddEqual(objs)
        if operator_id == 'subtract':
            op = task.SubtractEqual(objs)
        if operator_id == 'multiply':
            op = task.MultiplyEqual(objs)
        if operator_id == 'getshape':
            op = task.GetShape(objs)
        if operator_id == 'getcolor':
            op = task.GetColor(objs)
        if operator_id == 'iscolor':
            op = task.IsColor(objs)
        if operator_id == 'isshape':
            op = task.IsShape(objs)
        if operator_id == 'samecolor':
            op = task.SameColor(objs)
        if operator_id == 'sameshape':
            op = task.SameShape(objs)
        if operator_id == 'notsameshape':
            op = task.NotSameShape(objs)
        if operator_id == 'notsamecolor':
            op = task.NotSameColor(objs)
        if operator_id == 'sumeven':
            op = task.SumEven(objs)
        if operator_id == 'sumodd':
            op = task.SumOdd(objs)
        if operator_id == 'producteven':
            op = task.ProductEven(objs)
        if operator_id == 'productodd':
            op = task.ProductOdd(objs)

        if operator_id == 'switch':
            op = task.Switch

        return op

    def get_new_selectors(self):
        """
        Code to refresh task tree with new selector objects
        """
        all_selectors = taskgen.generate_select_sets(config.ALL_TIME,location=True,randomize=True)
        for node in self.DAG.nodes:
            op_id = self.DAG.nodes[node]['node_id']
            if op_id != 'switch':
                when = self.DAG.nodes[node]['selector_ids'][0]['when']
                if op_id in config.MULTI_OBJ_OPS:
                    num_selectors = np.random.choice([1,2])
                else:
                    num_selectors = 1
                #num_selectors = len(self.DAG.nodes[node]['selector_ids'])
                if num_selectors==2:
                    new_selector1 = random.choice(all_selectors[when])
                    new_selector2 = pick_2ndtask_selector(op_id,new_selector1)
                    self.DAG.nodes[node]['selector_ids'] = []
                    self.DAG.nodes[node]['selector_ids'].append(new_selector1)
                    self.DAG.nodes[node]['selector_ids'].append(new_selector2)
                else:
                    new_selector1 = random.choice(all_selectors[when])
                    self.DAG.nodes[node]['selector_ids'][0] = new_selector1

def generate_select_sets(whens, n_objects=None, randomize=True, location=False,nfeatures=0):
    """
    Generate a list of unique select instances to select specific objects during specific screen (time)
    
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

    if n_objects is None: 
        n_objects = len(all_colors)*len(all_shapes)

    # Create full list of locations
    if location:
        locations = []
        for x in range(config.GRIDSIZE_X):
            for y in range(config.GRIDSIZE_Y):
                locations.append((y,x))

        if randomize:
            random.shuffle(locations)

    # Create full list of locati    # Create full list of objects to sample from, which includes all possible color/shape combos
    all_colorshapes = []
    for color in all_colors:
        for shape in all_shapes:
            all_colorshapes.append(tuple((color,shape)))

    select_list = {}
    for when in set(whens):
        select_list[when] = []
        colorshapes_ind = list(range(len(all_colorshapes)))
        for i in range(n_objects):
            if randomize:
                tmp_colorshapes_ind = random.choice(colorshapes_ind) # randomly select colorshape index
                colorshapes_ind.remove(tmp_colorshapes_ind)
            else:
                tmp_colorshapes_ind = colorshapes_ind[i]
            #
            color = all_colorshapes[tmp_colorshapes_ind][0]
            shape = all_colorshapes[tmp_colorshapes_ind][1]
            #
            if location:
                for loc in locations:
                    obj_params = {'color':color,'shape':shape, 'when':when, 'loc':loc}
                    select_list[when].append(obj_params)
            else:
                obj_params = {'color':color,'shape':shape, 'when':when}
                select_list[when].append(obj_params)

    return select_list


def generate_select_sets_fortree(whens, n_objects=None, randomize=True,location=True,nfeatures=0):
    """
    Generate a list of unique select instances *that do not conflict* within a single tree 
    to select specific objects during specific screen (time)
    
    Args:
        n_objects       :       Number of objects to produce (if None, produce all possible combos)
        whens           :       list of available times 
        randomize (bool):       if selectors should have randomized ordering
        location        :       Include location attribute
        nfeatures         :       Number of features to select from each dimension (color, shape)
    """
    all_colors = config.ALL_COLORS.copy()
    all_shapes = config.ALL_SHAPES.copy()

    if nfeatures>0:
        all_colors = all_colors[:nfeatures] 
        all_shapes = all_shapes[:nfeatures]

    if n_objects is None: 
        n_objects = len(all_colors)*len(all_shapes)

    # Create full list of locations
    if location:
        locations = []
        for x in range(config.GRIDSIZE_X):
            for y in range(config.GRIDSIZE_Y):
                locations.append((y,x))

        if randomize:
            random.shuffle(locations)

    # Create full list of objects to sample from, which includes all possible color/shape combos
    all_colorshapes = []
    for color in all_colors:
        for shape in all_shapes:
            all_colorshapes.append(tuple((color,shape)))

    select_list = {}
    for when in set(whens):
        select_list[when] = []
        colorshapes_ind = list(range(len(all_colorshapes)))
        location_idx = 0
        for i in range(min(n_objects,len(locations))):
            if randomize:
                tmp_colorshapes_ind = random.choice(colorshapes_ind) # randomly select colorshape index
                colorshapes_ind.remove(tmp_colorshapes_ind)
            else:
                tmp_colorshapes_ind = colorshapes_ind[i]
            #
            color = all_colorshapes[tmp_colorshapes_ind][0]
            shape = all_colorshapes[tmp_colorshapes_ind][1]
            #
            if location:
                obj_params = {'color':color,'shape':shape, 'when':when, 'loc':locations[i]}
                select_list[when].append(obj_params)
            else:
                obj_params = {'color':color,'shape':shape, 'when':when}
                select_list[when].append(obj_params)

        #randomize again
        if randomize:
            random.shuffle(select_list[when])

    return select_list

def pick_2ndtask_selector(task,obj1,taken_locations=None):
    """"
    This function takes in the string of a task operator,
    and a dictionary select object, and returns a randomly chosen (appropriate)
    2nd select object that can be paired with the first object.
    This is required to avoid conflicting objects (e.g.,
    for the SumEven, we can't have two objects in the same location)
    """
    if task in ['sumeven','producteven','sumodd','productodd']:
        
        if 'loc' in obj1: loc1 = obj1['loc']
        if 'loc' in obj1:
            diffloc = False
            while diffloc is False:
                loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                if loc1!=loc2 and loc2 not in taken_locations:
                    diffloc = True
        col2 = random.choice(config.ALL_COLORS)
        shape2 = random.choice(config.ALL_SHAPES)
        obj2 = {}
        if 'loc' in obj1: obj2['loc'] = loc2
        obj2['color'] = col2
        obj2['shape'] = shape2
        obj2['when'] = obj1['when']

    if task in ['sameshape','notsameshape']:
        if 'loc' in obj1: loc1 = obj1['loc']
        shape1 = obj1['shape']
        if 'loc' in obj1:
            diffloc = False
            while diffloc is False:
                loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                if loc1!=loc2 and loc2 not in taken_locations:
                    diffloc = True
        
        shape2 = shape1
        # flip coin to see if it should be a different color
        chance = random.random()
        if chance > 0.5:
            while shape2==shape1:
                shape2 = random.choice(config.ALL_SHAPES)
        col2 = random.choice(config.ALL_COLORS)
        obj2 = {}
        if 'loc' in obj1: obj2['loc'] = loc2
        obj2['color'] = col2
        obj2['shape'] = shape2
        obj2['when'] = obj1['when']

    if task in ['samecolor','notsamecolor']:
        if 'loc' in obj1: loc1 = obj1['loc']
        color1 = obj1['color']
        if 'loc' in obj1:
            diffloc = False
            while diffloc is False:
                loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                if loc1!=loc2 and loc2 not in taken_locations:
                    diffloc = True
        
        color2 = color1
        # flip coin to see if it should be a different color
        chance = random.random()
        if chance > 0.5:
            while color2==color1:
                color2 = random.choice(config.ALL_COLORS)
        shape2 = random.choice(config.ALL_SHAPES)
        obj2 = {}
        if 'loc' in obj1: obj2['loc'] = loc2
        obj2['color'] = color2
        obj2['shape'] = shape2
        obj2['when'] = obj1['when']

    if task in ['existand','existor','existxor']:
        if 'loc' in obj1: loc1 = obj1['loc']
        color1 = obj1['color']
        shape1 = obj1['shape']
        if 'loc' in obj1:
            diffloc = False
            while diffloc is False:
                loc2 = (random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y)))
                if loc1!=loc2 and loc2 not in taken_locations:
                    diffloc = True
        
        color2 = random.choice(config.ALL_COLORS)
        shape2 = random.choice(config.ALL_SHAPES)
        # Can't be the same object
        while color1==color2 and shape1==shape2:
            color2 = random.choice(config.ALL_COLORS)
            shape2 = random.choice(config.ALL_SHAPES)

        obj2 = {}
        if 'loc' in obj1: obj2['loc'] = loc2
        obj2['color'] = color2
        obj2['shape'] = shape2
        obj2['when'] = obj1['when']

    return obj2



