#
import numpy as np
import random
import gcog.task.stim_generator as stim
import gcog.task.config as config

from collections import defaultdict
from collections import Counter
from copy import deepcopy

def obj_str(loc=None, color=None, shape=None, when=None, space_type=None):
	"""Get a string describing an object with attributes"""
	loc = loc or stim.Loc(None)
	color = color or stim.Color(None)
	shape = shape or stim.Shape(None)
	sentence = []

	if when is not None:
		sentence.append(when)
	if isinstance(color, stim.Attribute) and color.has_value:
		sentence.append(str(color))
	if isinstance(shape, stim.Attribute) and shape.has_value:
		sentence.append(str(shape))
	else:
		sentence.append('object')
	if isinstance(color, Operator):
		sentence += ['with', str(color)]
	if isinstance(shape, Operator):
		if isinstance(color, Operator):
			sentence.append('and')
		sentence += ['with', str(shape)]
	if isinstance(loc, Operator):
		sentence += ['on', space_type, 'of', str(loc)]
	return ' '.join(sentence)

class Skip(object):
    """Skip this operator)"""
    def __init__(self):
        pass

class Task(object):
    """Base class for tasks."""

    def __init__(self, operator=None, whens=None):
        if operator is None:
            self._operator = Operator()
        else:
            if not isinstance(operator, Operator):
                raise TypeError('operator is the wrong type ' + str(type(operator)))
            self._operator = operator

        # these vars are to ensure no conflicting colors emerge in objset for a task
        self.available_distractors_colors = None 
        self.available_distractors_shapes = None 

    def __call__(self, objset):
        return self._operator(objset)

    def __str__(self):
        return str(self._operator)

    def _get_all_nodes(self, op, visited):
        """Get the total number of operators in the graph starting with op."""
        visited[op] = True
        all_nodes = [op]
        for c in op.child:
            if isinstance(c, Operator) and not visited[c]:
                all_nodes.extend(self._get_all_nodes(c, visited))
        return all_nodes

    @property
    def _all_nodes(self):
        """Return all nodes in a list"""
        visited = defaultdict(lambda: False)
        return self._get_all_nodes(self._operator, visited)

    @property
    def whens(self):
        """Return a list of whens from which to sample from"""
        if self._whens:
            return self._whens
        else:
            raise Exception("whens not specified for this task yet")

    @whens.setter
    def whens(self, whens_list):
        self._whens = whens_list

    def operator_size(self):
        """Return the number of unique operators."""
        return len(self._all_nodes)

    def topological_sort_visit(self, node, visited, stack):
        """Recursive function that visits a node"""

        # mark the current as visited
        visited[node] = True

        # Recur for all the vertices adjacent to this vertex
        for child in node.child:
            if isinstance(child, Operator) and not visited[child]:
                self.topological_sort_visit(child, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, node)

    def topological_sort(self):
        """Perform a topological sort."""
        nodes = self._all_nodes

        # Mark all the vertices as not visited
        visited = defaultdict(lambda: False)
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for node in nodes:
            if not visited[node]:
                self.topological_sort_visit(node, visited, stack)

        return stack

    def guess_objset(self, objset, should_be=None):
        """
        Guesses object, and then adds it to the object set
        """
        nodes = self.topological_sort()
        should_be_dict = defaultdict(lambda: None)

        if should_be is not None:
            should_be_dict[nodes[0]] = should_be
        
        for node in nodes:
            should_be = should_be_dict[node]

            #should_be = node
            if isinstance(should_be, Skip):
                # check how many operators in child, then skip that number
                inputs = Skip() if len(node.child)==1 else [Skip()] * len(node.child)
            elif isinstance(node, Select):
                inputs = node.get_expected_input(should_be, objset)
                objset = inputs[0]
                inputs = inputs[1:]
            elif isinstance(node,(SameColor, SameShape, NotSameColor, NotSameShape)): 
                inputs = node.get_expected_input(should_be, objset)
            else:
                inputs = node.get_expected_input(should_be)

            if len(node.child) == 1:
                outputs = [inputs]
            else:
                outputs = inputs

            # Update the should_be dictionary for the node children
            for c, output in zip(node.child, outputs):
                if not isinstance(c, Operator):
                    continue
                if isinstance(output, Skip):
                    should_be_dict[c] = Skip()
                    continue
                if should_be_dict[c] is None:
                    # if not assigned, assign
                    should_be_dict[c] = output
                else:
                    # If assigned, for each object, try to merge them
                    if isinstance(c, Select):
                        # Loop over new output
                        for o in output:
                            assert isinstance(o, stim.Object)
                            merged = False
                            # Loop over previously assigned outputs
                            for s in should_be_dict[c]: 
                                # Try to merge
                                merged = s.merge(o)
                                if merged:
                                    break
                            if not merged:
                                should_be_dict[c].append(o)
                    else:
                        raise NotImplementedError()

        return objset

    def generate_objset(self, n_distractor=1):
        """
        Guess objset for all n_screen.
        Args:
            n_screen: list, list of screens 
            n_distracotr: int, number of distractors to add
        Returns:
            objset: full objset for all n_screens
        """

        objset = stim.ObjectSet(self.whens)

        for when in self.whens:
            for _ in range(n_distractor):
                objset.add_distractor(when,
                                      available_distractors_colors=self.available_distractors_colors,
                                      available_distractors_shapes=self.available_distractors_shapes)
        objset = self.guess_objset(objset)

        #### TODO?: Might want to reimplement this to make sure all screens have equal number of items
#        # Now make sure each screen has the same number of objects
#        max_distractors = 0
#        for when in objset.dict:
#            max_distractors = max(len(objset.dict[when]),max_distractors)
#        for when in objset.dict:
#            while len(objset.dict[when])<max_distractors:
#                objset.add_distractor(when)

        return objset

    def get_target(self, objset):
        return [self(objset)]

class Operator(object):
    """Base class for task operators."""

    def __init__(self):
        self.child = list()
        self.parent = list()

    
    def __call__(self, objset):
        del objset
        del screen_now

    def set_child(self, child):
        """Set operators as children."""
        try:
            child.parent.append(self)
            self.child.append(child)

        except AttributeError:
            for c in child:
                self.set_child(c)

class Select(Operator):
    """Selecting the objects that satisfy properties."""

    def __init__(self,
                 loc=None,
                 color=None,
                 shape=None,
                 when=None,
                 space_type=None):
        super(Select, self).__init__()

        loc = loc or stim.Loc(None)
        color = color or stim.Color(None)
        shape = shape or stim.Shape(None)

        # notimplemented, taku
        #if isinstance(loc, Operator) or loc.has_value:
        #    assert space_type is not None

        self.loc, self.color, self.shape = loc, color, shape
        self.set_child([loc, color, shape])
        self.when = when
        #self.space_type = space_type

    def __str__(self):
        """Get the language for color, shape, when combination."""
        #return obj_str(self.loc, self.color, self.shape, self.when, self.space_type)
        return obj_str(self.loc, self.color, self.shape, self.when)

    def __call__(self, objset):
        """Return a subset of objset."""
        loc = self.loc(objset)
        color = self.color(objset)
        shape = self.shape(objset)
        #space = self.space(objset, screen_now)

        #space = loc.get_space_to(self.space_type)

        subset = objset.select(self.when,
                               color=color,
                               shape=shape)

        return subset

    def get_expected_input(self, should_be, objset):
        """
        Guess objset for Select operator.
        Optionally modify the objset based on the target output, should_be, and pass down the supposed input attributes
        There are two sets of attributes to compute:
        (1) The attributes of the expected input, i.e., the attributes to select
        (2) The attributes of new object being added to the objset
        There are two main scenarios:
        (1) The target output is empty
        (2) the target output is not empty, then it has to be a list with a single Object instnace in it
        When (1) the target output is empty, the goal is to make sure attr_newobject and attr_expectedinput differ by a single attribute.
        When (2) the target output is not empty, then attr_newobject and attr_expectedinput are the same.
        We first decide the attributes of the expected inputs.
        If target output is empty, then expected inputs is determined solely based on its actual inputs. If the actual inputs can be computed, then use it, and skip the traversal of following nodes. Otherwise, use a random allowed value.
        If target output is not empty, again, if the actual inputs can be computed, use it. If the actual input can not be computed, but the attribute is specificed by the target uoutput, use that. Otherwise, use a random value.
        Then determine the new objsect being added.
        ### This is not implemented; a response will always happen (given a switch condition).
        If target output is empty, hten randomly change one of the attributes taht is not None.
        For self.space attribute, if it is an operator, then add the object at the opposite side of space #TODO not implemented
        If target output is not empty, use the attributes of the expected input
        Args:
            should_be: a list of Object instances
            objset: objset instance
        Returns:
            objset: objset instance
            space: supposed input # not implemented
            color: supposed input
            shape: supposed input
        Raises:
            TypeError: when should_be is not a list of Object instances
            NotImplementedError: when should_be is a list and has length > 1
        """

        if should_be is None:
            raise NotImplementedError('target attributes are None')

        if should_be:
            #Make sure should_be is a length-1 list of Object instance
            for s in should_be:
                if not isinstance(s, stim.Object):
                    raise TypeError('Wrong type in should_be list: ' + str(type(s)))

            #if len(should_be) > 1:
            #    for s in should_be:
            #        print(s)
            #    raise NotImplementedError()
            obj_should_be = should_be[0]

            attr_new_object = list()
            attr_type_not_fixed = list()
            # first evaluate the inputs taht can be evaluated
            #for attr_type in ['loc', 'color', 'shape']:
            for attr_type in ['color', 'shape','loc']:
                # get select attributes
                a = getattr(self, attr_type)
                attr = a(objset, self.when)
                # get object attributes that may have been passed by get_expected_node
                obj_attr = getattr(obj_should_be, attr_type)
                # If hte input is successfullly evaluated
                if attr.has_value:
                    #if attr_type == 'loc':
                    #    raise NotImplementedError()
                    #    attr = attr.get_space_to(self.space_type)
                    attr_new_object.append(attr)
                elif obj_attr.has_value:
                    # add attribute if should_be object has specified attribute
                    attr_new_object.append(obj_attr) 
                else:
                    # Find attributes that are okay to use (non-conflicting)
                    attr_type_not_fixed.append(attr_type)

            # Add an object based on these attributes
            # Note that, objset.add() will implicitly select the objset
            obj = stim.Object(attr_new_object, when=self.when)
            obj = objset.add(obj, add_if_exist=False)

            if obj is None:
                return objset, Skip(), Skip(), Skip()

            # If some attributes of the object are given by should_be, use them
            for attr_type in attr_type_not_fixed:
                a = getattr(obj_should_be, attr_type)
                if a.has_value:
                    setattr(obj, attr_type, a) # set obj.a = attr_type

            # If an attribute of select is an operator, then the expected input is the value of obj
            attr_expected_in = list()
            for attr_type in ['loc', 'color', 'shape']:
                a = getattr(self, attr_type)
                if isinstance(a, Operator):
                    # only operators need expected_in
                    if attr_type == 'loc':
                        raise NotImplementedError()
                        space = obj.loc.get_opposite_space_to(self.space_type)
                        attr = space.sample()
                    else:
                        attr = getattr(obj, attr_type)
                    attr_expected_in.append(attr)
                else:
                    attr_expected_in.append(Skip())

        if not should_be:
            # First determine the attributes to flip later
            attr_type_to_flip = list()
            for attr_type in ['loc', 'color', 'shape']:
                a = getattr(self, attr_type)
                # if attribute is operator or is a specified value
                if isinstance(a, Operator) or a.has_value:
                    attr_type_to_flip.append(attr_type)

            # Now generate expected input attributes
            attr_expected_in = list()
            attr_new_object = list()
            for attr_type in ['loc', 'color', 'shape']:
                a = getattr(self, attr_type)
                attr = a(objset)
                if isinstance(a, Operator):
                    if attr == 'INVALID':
                        # Can't be evaluated yet, then radomly choose one
                        attr = stim.random_attr(attr_type)
                    attr_expected_in.append(attr)
                else:
                    attr_expected_in.append(Skip())

                if attr_type in attr_type_to_flip:
                    # Candidate attribute values for the new object
                    attr_new_object.append(attr)

            # Randomly pick one attribute to flip
            attr_type = random.choice(attr_type_to_flip)
            i = attr_type_to_flip.index(attr_type)
            if attr_type == 'loc':
                raise NotImplementedError() # Taku
            else:
                # Select a different attribute
                attr_new_object[i] = stim.another_attr(attr_new_object[i])
                # not flipping loc, so place it in the correct direction
                if 'loc' in attr_type_to_flip:
                    raise NotImplementedError() # taku

            # Add an object based on these attributes
            obj = stim.Object(attr_new_object, when=self.when)
            obj = objset.add(obj, add_if_exist=False)

        return [objset] + attr_expected_in

class Get(Operator):
    """Get attribute of an object."""

    def __init__(self, attr_type, objs):
        """ Get attribute of an object.
        Args:
            attr_type: string, color, shape, or loc. The type of attribute to get.
            objs: Operator instance or Object instance
        """
        super(Get,self).__init__()
        self.attr_type = attr_type
        self.objs = objs
        assert isinstance(objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        if attr_type == 'color':
            self.objs.color.value = None
        elif attr_type == 'shape':
            self.objs.shape.value = None

        self.set_child(objs)

    def __str__(self):
        words = [self.attr_type, 'of', str(self.objs)]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
            scree_now: current screen
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            objs = self.objs(objset)
            return getattr(objs[0], self.attr_type)
        else:
            objs = self.objs

        if len(objs) != 1:
            return 'INVALID'
        else:
            return getattr(objs[0], self.attr_type)

    def operator_length(self):
        return 1 # just a single operation for get


    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = stim.random_attr(self.attr_type)
        objs = stim.Object([should_be])
        return [objs]

class GetShape(Operator):
    """Get shape of an object."""

    def __init__(self, objs):
        """ Get shape of an object.
        Args:
            objs: Operator instance or Object instance
        """
        super(GetShape,self).__init__()
        self.attr_type = 'shape'
        self.objs = objs[0]
        assert isinstance(self.objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        if self.objs.shape.has_value:
            self.shouldbeattr = deepcopy(self.objs.shape)
        self.objs.shape.value = None

        self.set_child(objs)

    def __str__(self):
        words = [self.attr_type, 'of', str(self.objs)]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            objs = self.objs(objset)
            return getattr(objs[0], self.attr_type)
        else:
            objs = self.objs

        if len(objs) != 1:
            return 'INVALID'
        else:
            return getattr(objs[0], self.attr_type)

    def operator_length(self):
        return 1 # just a single operation for get


    def get_expected_input(self, should_be):
        if should_be is None:
            if self.shouldbeattr:
                should_be = self.shouldbeattr
            else:
                should_be = stim.random_attr(self.attr_type)
        objs = stim.Object([should_be])
        return [objs]

class GetColor(Operator):
    """Get attribute of an object."""

    def __init__(self, objs):
        """ Get Color of an object.
        Args:
            objs: Operator instance or Object instance
        """
        super(GetColor,self).__init__()
        self.attr_type = 'color'
        self.objs = objs[0]
        assert isinstance(self.objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        if self.objs.color.has_value:
            self.shouldbeattr = deepcopy(self.objs.color)
        self.objs.color.value = None

        self.set_child(objs)

    def __str__(self):
        words = [self.attr_type, 'of', str(self.objs)]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
            scree_now: current screen
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            objs = self.objs(objset)
            return getattr(objs[0], self.attr_type)
        else:
            objs = self.objs

        if len(objs) != 1:
            return 'INVALID'
        else:
            return getattr(objs[0], self.attr_type)

    def operator_length(self):
        return 1 # just a single operation for get


    def get_expected_input(self, should_be):
        if should_be is None:
            if self.shouldbeattr:
                should_be = self.shouldbeattr
            else:
                should_be = stim.random_attr(self.attr_type)
        objs = stim.Object([should_be])
        return [objs]

class IsColor(Operator):
    """Is this shape this color?"""

    def __init__(self, objs):
        """ Is this shape this color? 
        Args:
            objs: Operator instance or Object instance
        """
        super(IsColor,self).__init__()
        self.attr_type = 'color'
        self.objs = objs[0]
        assert isinstance(self.objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        self.tobeattr = deepcopy(self.objs.color)
        self.objs.color.value = None

        self.set_child(objs)

    def __str__(self):
        words = ['is the', str(self.objs), 'the', self.attr_type, self.tobeattr.value]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
            scree_now: current screen
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            subset = self.objs(objset)

            if subset[0].color.value == self.tobeattr.value:
                return True
            else:
                return False

    def operator_length(self):
        return 1 # just a single operation for get


    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random()>0.5

        if should_be:
            should_be = self.tobeattr
        else:
            should_be = self.tobeattr
            while should_be==self.tobeattr:
                should_be = stim.random_attr(self.attr_type)

        objs = stim.Object([should_be])
        return [objs]

class IsShape(Operator):
    """Get attribute of an object."""

    def __init__(self, objs):
        """ Is this color this shape? 
        Args:
            objs: Operator instance or Object instance
        """
        super(IsShape,self).__init__()
        self.attr_type = 'shape'
        self.objs = objs[0]
        assert isinstance(self.objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        self.tobeattr = deepcopy(self.objs.shape)
        self.objs.shape.value = None

        self.set_child(objs)

    def __str__(self):
        words = ['is', str(self.objs), self.tobeattr.value]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
            scree_now: current screen
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            subset = self.objs(objset)

            if subset[0].shape.value == self.tobeattr.value:
                return True
            else:
                return False

    def operator_length(self):
        return 1 # just a single operation for get


    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random()>0.5

        if should_be:
            should_be = self.tobeattr
        else:
            should_be = self.tobeattr
            while should_be==self.tobeattr:
                should_be = stim.random_attr(self.attr_type)

        objs = stim.Object([should_be])
        return [objs]

class Go(Operator):
    """Get attribute of an object."""

    def __init__(self, objs):
        """ Get Location of an object.
        Args:
            objs: Operator instance or Object instance
        """
        super(Go,self).__init__()
        self.attr_type = 'loc'
        self.objs = objs
        assert isinstance(self.objs, Operator)
        # now make sure obj attr is None (so we can reset it)
        if self.objs.loc.has_value:
            self.shouldbeattr = deepcopy(self.objs.loc)
        else:
            self.shouldbeattr = stim.random_loc()
        self.objs.loc.value = None

        self.set_child(objs)

    def __str__(self):
        words = [self.attr_type, 'of', str(self.objs)]
        if not self.parent:
            words += ['?'] 
        return ' '.join(words)

    def __call__(self, objset):
        """Get the attribute.
        By default, get the attribute of the unique object. If there are multiple objects, then return INVALID.
        Args:
            objset: objset
            scree_now: current screen
        Returns:
            attr: Attribute instance of INVALID
        """
        if isinstance(self.objs, Operator):
            objs = self.objs(objset)
            return getattr(objs[0], self.attr_type)
        else:
            objs = self.objs

        if len(objs) != 1:
            return 'INVALID'
        else:
            return getattr(objs[0], self.attr_type)

    def operator_length(self):
        return 1 # just a single operation for get

    def get_expected_input(self, should_be):
        if should_be is None:
            if self.shouldbeattr:
                should_be = self.shouldbeattr
            else:
                should_be = stim.random_attr(self.attr_type)
        objs = stim.Object([should_be])
        return [objs]

class Exist(Operator):
    """Check if object with property exists"""

    def __init__(self, objs):
        super(Exist, self).__init__()
        self.objs = objs
        assert isinstance(objs, Operator)
        self.set_child(objs)

    def __str__(self):
        words = [str(self.objs), 'exist']
        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        subset = self.objs(objset)
        if subset == 'INVALID':
            return 'INVALID'
        elif subset:
            # if subset is not empty
            return True
        else:   
            return False

    def get_expected_input(self, should_be):

        # Randomly determine if this item exists or not
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = [stim.Object()]
        else:
            # if statement is false, then skip child nodes/operators
            should_be = Skip()

        return should_be

    def operator_length(self):
        return 2 # exist requires select + exist 

class ExistAnd(Operator):
    """Check if object with property exists"""

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(ExistAnd, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['and']
            counter += 1
        words += ['exist']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        exist = True
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'
            if subset:
                exist *= True
            else:
                exist *= False

        statement = True if exist else False
        
        return statement

    def operator_length(self):
        return 1 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        # Randomly determine if this item exists or not
        if should_be is None:
            should_be = random.random() > 0.5

        num_exist_op = len(self.objs)
        if should_be:
            should_be = []
            for _ in range(num_exist_op): should_be.append([stim.Object()])
        else:
            # if operator is false, at least randomly select some objects do exist (as to make the task not too easy)
            should_be = []
            for o in self.objs:
                if random.random()>0.5: 
                    should_be.append([stim.Object()])
                else:
                    should_be.append(Skip())

            # Ensure that not all objects exist (this would make the statement true)
            skip_count = 0
            for o in should_be:
                if isinstance(o, Skip):
                    skip_count += 1
            # if no skips found, randomly choose an object to be a skip
            if skip_count==0: 
                rand_int = random.choice(range(num_exist_op))
                should_be[rand_int] = Skip()

        return should_be

class ExistOr(Operator):
    """Check if object with property exists - if A or B or C exists"""

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(ExistOr, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

    def __str__(self):
        #words = [for o in self.objs: str(o), 'and', str(self. bjs), 'exist']
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['or']
            counter += 1
        words += ['exist']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        exist = False
        for select in self.objs:
            #for o in objset: print(o)
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'
            if subset:
                exist = True

        statement = True if exist else False
        
        return statement
        
    def get_expected_input(self, should_be):
        # Randomly determine if this item exists or not
        if should_be is None:
            should_be = random.random() > 0.5

        num_exist_op = len(self.objs)
        if should_be:
            # Then OR is True; not all elements need to exist
            should_be = []
            for _ in range(num_exist_op): 
                if random.random()>0.5:
                    should_be.append([stim.Object()])
                else:
                    should_be.append(Skip())

            # Now ensure that at least one object exists to satisfy the statement
            skip_count = 0
            for o in should_be:
                if isinstance(o, Skip):
                    skip_count += 1
            # if no skips found, randomly choose an object to be a skip
            if skip_count==num_exist_op: 
                rand_int = random.choice(range(num_exist_op))
                should_be[rand_int] = [stim.Object()]
        else:
            # if operator is false, then everything should be false
            should_be = []
            for o in self.objs:
                should_be.append(Skip())

        return should_be

    def operator_length(self):
        return 1 + len(self.objs) # exist requires select + exist 

class ExistXor(Operator):
    """Check if object with property exists - if A xor B xor C exists (i.e., only one is true)"""

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(ExistXor, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

    def __str__(self):
        #words = [for o in self.objs: str(o), 'and', str(self. bjs), 'exist']
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['xor']
            counter += 1
        words += ['exist']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        exist = 0
        for select in self.objs:
            #for o in objset: print(o)
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'
            if subset:
                exist += 1  

        statement = True if exist==1 else False
        
        return statement
        
    def get_expected_input(self, should_be):

        # Randomly determine if this item exists or not
        if should_be is None:
            should_be = random.random() > 0.5

        num_exist_op = len(self.objs)
        if should_be:
            # Then XOR is True; only 1 element is True 
            should_be = []
            for o in self.objs:
                should_be.append(Skip())
            rand_int = random.choice(range(num_exist_op))
            should_be[rand_int] = [stim.Object()]
        else:
            # IF not true, then 0 or more than 1 element should exist
            should_be = []
            for _ in range(num_exist_op): 
                if random.random()>0.5:
                    should_be.append([stim.Object()])
                else:
                    should_be.append(Skip())

            # Now ensure that at least one object exists to satisfy the statement
            skip_count = 0
            for o in should_be:
                if isinstance(o, Skip):
                    skip_count += 1
            # if only one non-skip is found, then flip a coin. if heads then add another object, else remove object 
            if skip_count==num_exist_op-1: 
                if random.random()>0.5:
                    rand_int = random.choice(range(num_exist_op))
                    for i in range(len(should_be)):
                        if isinstance(should_be[i], Skip):
                            should_be[i] = [stim.Object()]
                else:
                    should_be = []
                    for _ in range(num_exist_op):
                        should_be.append(Skip())

        return should_be

    def operator_length(self):
        return 1 + len(self.objs) # exist requires select + exist 

class Switch(Operator):
    """Switch behaviors based on trueness of statement, i.e., if-then -> else
    Args:
        statement: boolean type operator
        do_if_true: operator performed it the evaluated statement is True
        do_if_false: operator performed if the evluated statement is False
        invalid_as_false: When True, invalid statement is treated as FAlse
        both_options_avail: When True, both do_if_true and do_if_false will be called the get_expected_input function
    """

    def __init__(self,
                 statement,
                 do_if_true,
                 do_if_false,
                 invalid_as_false=False,
                 both_options_avail=True):
        super(Switch, self).__init__()

        self.statement = statement
        self.do_if_true = do_if_true
        self.do_if_false = do_if_false

        self.set_child([statement, do_if_true, do_if_false])

        self.invalid_as_false = invalid_as_false
        self.both_options_avail = both_options_avail

    def __str__(self):
        # Edit -- allow nested switch statements, even if they're not english readable

        words = [
                '[ if',
                str(self.statement), ',', 'then',
                str(self.do_if_true), ',', 'else',
                str(self.do_if_false), ']'
                ]
        return ' '.join(words)

    def __call__(self, objset):
        statement_true = self.statement(objset)
        if statement_true == 'INVALID':
            if self.invalid_as_false:
                statement_true = False
            else:
                return 'INVALID'

        if statement_true: 
            return self.do_if_true(objset)
        else:
            return self.do_if_false(objset)

    def __getattr__(self, key):
        """Get attributes.
        The swithc operator should also have the shared attributes of do_if_true and do_if_false. 
        Because regardless of the statement truthness, the common attributes will be true.
        Args: 
            key: attribute
        Returns: 
            attr1: common attribute of do_if_true and do_if_false
        Raises:
            ValueError: if the attribute is not common. #TODO
        """

        attr1 = getattr(self.do_if_true, key)
        attr2 = getattr(self.do_if_false, key)
        if attr1==attr2:
            return attr1
        else:
            raise ValueError()

    def get_expected_input(self, should_be=None):
        """
        Guess objset for Switch operator.
        Here the objset is guessed based on whether or not the statement should be true.
        Both do_if_true and do_if_false should be executable regardless of the statement truthfulness.
        In this way, it doesn't provide additional information.
        """
        if should_be is None:
            should_be = random.random() > 0.5 # randomly select which conditional is true
        return should_be, None, None

    def operator_length(self):
        return 1 

class SameColor(Operator):
    """Check if two color attributes are the same."""

    def __init__(self, objs):
        """Compare to attributes.
        Args:
            objs: list of select operators (to make consistent with other task ops)
        """
        super(SameColor, self).__init__()
        self.attr_type = 'color'
        for o in objs:
            # set it to None so it can be determined by operator
            o.color.value = None 

        if len(objs)>len(config.ALL_SHAPES):
            raise Exception("Number of comparable colors cannot exceed number of unique color attributes")

        # now make sure the object attributes are not the same for other attribute (would make task ill-posed)
        all_attr = []
        for o in objs:
            all_attr.append(o.shape.value)
        
        counter_dict = Counter(all_attr) # dict that counts occurrences of each attribute
        available_attrs = config.ALL_SHAPES.copy()
        for attr in counter_dict: 
            if attr is not None:
                available_attrs.remove(attr)
        for o in objs:
            if counter_dict[o.shape.value] > 1:
                o.shape.value = random.choice(available_attrs)
                available_attrs.remove(o.shape.value)
                counter_dict[o.shape.value] -= 1

        # Now transform select_operators into get operators
        gets = []
        for o in objs:
            gets.append(Get(self.attr_type, o))

        self.gets = gets

        self.set_child(gets)

    def __str__(self): 
        words = []
        counter = 0
        for get in self.gets:
            words += [str(get)]
            if counter < len(self.gets)-1: words += ['and']
            counter += 1
        words += ['same']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        issame = True
        attrs = []
        for get in self.gets:
            attrs.append(get(objset)) 

        # now compare first attribute with all other attributes, to make sure they're the same
        attr1 = attrs[0]
        for attr in attrs[1:]:
            issame *= attr1.value == attr.value

        statement = True if issame else False
        return statement

    def get_expected_input(self, should_be, objset):
        #Determine which attribute should be fixed and which shouldn't

        if should_be is None:
            should_be = random.random() > 0.5

        num_gets = len(self.gets)

        objs = []
        if should_be:
            should_be = stim.random_attr(self.attr_type)
            for _ in range(num_gets):
                objs.append(should_be)
        else:
            attr1 = stim.random_attr(self.attr_type)
            objs.append(attr1)
            for _ in range(num_gets-1):
                objs.append(stim.another_attr(attr1))

        return objs

    def operator_length(self):
        return 1 + len(self.gets) # get + select n times 

class SameShape(Operator):
    """Check if two shape attributes are the same."""

    def __init__(self, objs):
        """Compare to attributes.
        Args:
            objs: list of select operators (to make consistent with other task ops)
        """
        super(SameShape, self).__init__()
        self.attr_type = 'shape'
        for o in objs:
            # set it to None so it can be determined by operator
            o.shape.value = None 

        if len(objs)>len(config.ALL_SHAPES):
            raise Exception("Number of comparable shapes cannot exceed number of unique shape attributes")

        # now make sure the object attributes are not the same for other attribute (would make task ill-posed)
        all_attr = []
        for o in objs:
            all_attr.append(o.color.value)
        
        counter_dict = Counter(all_attr) # dict that counts occurrences of each attribute
        available_attrs = config.ALL_COLORS.copy()
        for attr in counter_dict: 
            if attr is not None: 
                available_attrs.remove(attr)
        for o in objs:
            if counter_dict[o.color.value] > 1:
                o.color.value = random.choice(available_attrs)
                available_attrs.remove(o.color.value)
                counter_dict[o.color.value] -= 1

        # Now transform select_operators into get operators
        gets = []
        for o in objs:
            gets.append(Get(self.attr_type, o))

        self.gets = gets

        self.set_child(gets)

    def __str__(self): 
        words = []
        counter = 0
        for get in self.gets:
            words += [str(get)]
            if counter < len(self.gets)-1: words += ['and']
            counter += 1
        words += ['same']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        issame = True
        attrs = []
        for get in self.gets:
            attrs.append(get(objset)) 

        # now compare first attribute with all other attributes, to make sure they're the same
        attr1 = attrs[0]
        for attr in attrs[1:]:
            issame *= attr1.value == attr.value

        statement = True if issame else False
        return statement

    def get_expected_input(self, should_be, objset):
        #Determine which attribute should be fixed and which shouldn't

        if should_be is None:
            should_be = random.random() > 0.5

        num_gets = len(self.gets)

        objs = []
        if should_be:
            should_be = stim.random_attr(self.attr_type)
            for _ in range(num_gets):
                objs.append(should_be)
        else:
            attr1 = stim.random_attr(self.attr_type)
            objs.append(attr1)
            for _ in range(num_gets-1):
                objs.append(stim.another_attr(attr1))

        return objs

    def operator_length(self):
        return 1 + len(self.gets) # get + select n times 

class NotSameColor(Operator):
    """Check if two color attributes are not the same."""

    def __init__(self, objs):
        """Compare to attributes.
        Args:
            objs: list of select operators (to make consistent with other task ops)
        """
        super(NotSameColor, self).__init__()
        self.attr_type = 'color'
        for o in objs:
            # set it to None so it can be determined by operator
            o.color.value = None 

        if len(objs)>len(config.ALL_SHAPES):
            raise Exception("Number of comparable colors cannot exceed number of unique color attributes")

        # now make sure the object attributes are not the same for other attribute (would make task ill-posed)
        all_attr = []
        for o in objs:
            all_attr.append(o.shape.value)
        
        counter_dict = Counter(all_attr) # dict that counts occurrences of each attribute
        available_attrs = config.ALL_SHAPES.copy()
        for attr in counter_dict: 
            if attr is not None:
                available_attrs.remove(attr)
        for o in objs:
            if counter_dict[o.shape.value] > 1:
                o.shape.value = random.choice(available_attrs)
                available_attrs.remove(o.shape.value)
                counter_dict[o.shape.value] -= 1

        # Now transform select_operators into get operators
        gets = []
        for o in objs:
            gets.append(Get(self.attr_type, o))

        self.gets = gets

        self.set_child(gets)

    def __str__(self): 
        words = []
        counter = 0
        for get in self.gets:
            words += [str(get)]
            if counter < len(self.gets)-1: words += ['and']
            counter += 1
        words += ['not same']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        issame = True
        attrs = []
        for get in self.gets:
            attrs.append(get(objset)) 

        # now compare first attribute with all other attributes, to make sure they're the same
        attr1 = attrs[0]
        for attr in attrs[1:]:
            issame *= attr1.value == attr.value

        statement = False if issame else True
        return statement

    def get_expected_input(self, should_be, objset):
        #Determine which attribute should be fixed and which shouldn't
        if should_be is None:
            should_be = random.random() > 0.5

        num_gets = len(self.gets)

        objs = []
        if should_be:
            attr1 = stim.random_attr(self.attr_type)
            objs.append(attr1)
            for _ in range(num_gets-1):
                objs.append(stim.another_attr(attr1))
        else:
            should_be = stim.random_attr(self.attr_type)
            for _ in range(num_gets):
                objs.append(should_be)

        return objs

    def operator_length(self):
        return 2 + len(self.gets) # get + select n times 

class NotSameShape(Operator):
    """Check if two shape attributes are NOT the same."""

    def __init__(self, objs):
        """Compare to attributes.
        Args:
            objs: list of select operators (to make consistent with other task ops)
        """
        super(NotSameShape, self).__init__()
        self.attr_type = 'shape'
        for o in objs:
            # set it to None so it can be determined by operator
            o.shape.value = None 

        if len(objs)>len(config.ALL_SHAPES):
            raise Exception("Number of comparable shapes cannot exceed number of unique shape attributes")

        # now make sure the object attributes are not the same for other attribute (would make task ill-posed)
        all_attr = []
        for o in objs:
            all_attr.append(o.color.value)
        
        counter_dict = Counter(all_attr) # dict that counts occurrences of each attribute
        available_attrs = config.ALL_COLORS.copy()
        for attr in counter_dict: 
            if attr is not None:
                available_attrs.remove(attr)
        for o in objs:
            if counter_dict[o.color.value] > 1:
                o.color.value = random.choice(available_attrs)
                available_attrs.remove(o.color.value)
                counter_dict[o.color.value] -= 1

        # Now transform select_operators into get operators
        gets = []
        for o in objs:
            gets.append(Get(self.attr_type, o))

        self.gets = gets

        self.set_child(gets)

    def __str__(self): 
        words = []
        counter = 0
        for get in self.gets:
            words += [str(get)]
            if counter < len(self.gets)-1: words += ['and']
            counter += 1
        words += ['not same']

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        issame = True
        attrs = []
        for get in self.gets:
            attrs.append(get(objset)) 

        # now compare first attribute with all other attributes, to make sure they're the same
        attr1 = attrs[0]
        for attr in attrs[1:]:
            issame *= attr1.value == attr.value

        statement = False if issame else True
        return statement

    def get_expected_input(self, should_be, objset):
        #Determine which attribute should be fixed and which shouldn't
        if should_be is None:
            should_be = random.random() > 0.5

        num_gets = len(self.gets)

        objs = []
        if should_be:
            attr1 = stim.random_attr(self.attr_type)
            objs.append(attr1)
            for _ in range(num_gets-1):
                objs.append(stim.another_attr(attr1))
        else:
            should_be = stim.random_attr(self.attr_type)
            for _ in range(num_gets):
                objs.append(should_be)

        return objs

    def operator_length(self):
        return 2 + len(self.gets) # get + select n times 

class ProductEven(Operator):
    """
    Multiplies object location values and asks if product is even
    For a single object, multiplies its x * y location
    For multiple objects:
    obj1.x * obj1.y * obj2.x + obj2.y

    """

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(ProductEven, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['*']
            counter += 1
        words += ['product is even'] 

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 1
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result *= subset[0].loc.value[0] * subset[0].loc.value[1]
        
        statement = True if result%2==0 else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = False
            locs = []
            result = 1
            while result%2!=0: # while odd
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result *= loc.value[0] * loc.value[1]
                    locs.append(loc)
        else:
            result = 2
            while result%2==0: # while even, make it odd
                result = 1
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result *= loc.value[0] * loc.value[1]
                    locs.append(loc)

        should_be = []
        for loc in locs:
            should_be.append([stim.Object([loc])])

        if len(locs)==1:
            should_be = should_be[0]

        return should_be

class ProductOdd(Operator):
    """
    Multiplies object location values and asks if product is odd
    For a single object, multiplies its x * y location
    For multiple objects:
    obj1.x * obj1.y * obj2.x + obj2.y

    """

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(ProductOdd, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['*']
            counter += 1
        words += ['product is odd'] 

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 1
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result *= subset[0].loc.value[0] * subset[0].loc.value[1]
        
        statement = True if result%2==1 else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = False
            locs = []
            result = 0
            while result%2==0: # while even; wait till odd
                result = 1 # reset to 1
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result *= loc.value[0] * loc.value[1]
                    locs.append(loc)
        else:
            result = 1
            while result%2==1: # while odd, wait till even 
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result *= loc.value[0] * loc.value[1]
                    locs.append(loc)

        should_be = []
        for loc in locs:
            should_be.append([stim.Object([loc])])

        if len(locs)==1:
            should_be = should_be[0]

        return should_be

class SumEven(Operator):
    """
    Add the location elements of an object and ask if sum is even
    Ex: If obj is at (x,y) location, then return x + y.
    For multiple objects:

    + obj1.x + obj1.y + obj2.x + obj2.y
    """
    
    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(SumEven, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs
        self.when = objs[0].when

        self.set_child(objs)

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['+']
            counter += 1
        words += ['sum is even'] 

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 0
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result += subset[0].loc.value[0] + subset[0].loc.value[1]
        
        statement = True if result%2==0  else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = []
            result = 1 # first assume odd
            while result%2!=0:
                result = 0 # reset to 0
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result += loc.value[0] + loc.value[1]
                    locs.append(loc)
        else:
            result = 0 # assume even
            while result%2==0:
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result += loc.value[0] + loc.value[1]
                    locs.append(loc)

        should_be = []
        for loc in locs:
            should_be.append([stim.Object([loc])])

        if len(locs)==1:
            should_be = should_be[0]

        return should_be

class SumOdd(Operator):
    """
    Add the location elements of an object and ask if sum is odd
    Ex: If obj is at (x,y) location, then return x + y.
    For multiple objects:

    + obj1.x + obj1.y + obj2.x + obj2.y
    """
    
    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(SumOdd, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs
        self.when = objs[0].when

        self.set_child(objs)

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['+']
            counter += 1
        words += ['sum is odd'] 

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 0
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result += subset[0].loc.value[0] + subset[0].loc.value[1]
        
        statement = True if result%2==1  else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = []
            result = 2 # first assume even
            while result%2==0:
                result = 0 # reset to 0
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result += loc.value[0] + loc.value[1]
                    locs.append(loc)
        else:
            result = 1 # assume odd
            while result%2==1:
                result = 0 # reset to 0
                locs = []
                for _ in range(len(self.objs)):
                    loc = stim.random_loc()
                    result += loc.value[0] + loc.value[1]
                    locs.append(loc)

        should_be = []
        for loc in locs:
            should_be.append([stim.Object([loc])])

        if len(locs)==1:
            should_be = should_be[0]

        return should_be



##### UNUSED
class AddEqual(Operator):
    """
    Add the location elements of an object
    Ex: If obj is at (x,y) location, then return x + y.
    For multiple objects:

    + obj1.x + obj1.y + obj2.x + obj2.y
    """
    
    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(AddEqual, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs
        self.when = objs[0].when

        self.set_child(objs)

        # set the question value
        question = 0
        locs = []
        for o in self.objs:
            loc = stim.random_loc()
            question += loc.value[0] + loc.value[1]
            locs.append(loc)
        self.question = question
        self.locs = locs

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['+']
            counter += 1
        words += ['sum equals'] 
        words += [str(self.question)]

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 0
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result += subset[0].loc.value[0] + subset[0].loc.value[1]
        
        statement = True if self.question==result else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = []
            for loc in self.locs:
                should_be.append([stim.Object([loc])])
        else:
            result = self.question
            while result==self.question:
                result = 0
                should_be = []
                for _ in self.locs:
                    loc = stim.random_loc()
                    result += loc.value[0] + loc.value[1]
                    should_be.append([stim.Object([loc])])

        if len(self.locs)==1:
            should_be = should_be[0]

        return should_be

class SubtractEqual(Operator):
    """
    Subtract objects using numeric representation of shapes
    for objects, compute the locations y-x
    - obj1.x - obj1.y - obj2.x - obj2.y
    """

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(SubtractEqual, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

        # set the question value
        question = 0
        locs = []
        for o in self.objs:
            loc = stim.random_loc()
            question += - loc.value[0] - loc.value[1]
            locs.append(loc)
        self.question = question
        self.locs = locs

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['-']
            counter += 1
        words += ['difference equals'] 
        words += [str(self.question)]

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 0
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result += -subset[0].loc.value[0] - subset[0].loc.value[1]
        
        statement = True if self.question==result else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = []
            for loc in self.locs:
                should_be.append([stim.Object([loc])])
        else:
            result = self.question
            while result==self.question:
                result = 0
                should_be = []
                for _ in self.locs:
                    loc = stim.random_loc()
                    result += - loc.value[0] - loc.value[1]
                    should_be.append([stim.Object([loc])])

        if len(self.locs)==1:
            should_be = should_be[0]

        return should_be

class MultiplyEqual(Operator):
    """
    Multiplies object location values
    For a single object, multiplies its x * y location
    For multiple objects:
    obj1.x * obj1.y * obj2.x + obj2.y

    """

    def __init__(self, objs):
        """
        Args:
            objs    :   List of Select operators
        """
        super(MultiplyEqual, self).__init__()
        for obj in objs:
            assert isinstance(obj, Operator)
        self.objs = objs

        self.set_child(objs)

        # set the question value
        question = 1
        locs = []
        for o in self.objs:
            loc = stim.random_loc()
            question *= loc.value[0] * loc.value[1]
            locs.append(loc)
        self.question = question
        self.locs = locs

    def __str__(self):
        words = []
        counter = 0
        for o in self.objs:
            words += [str(o)]
            if counter < len(self.objs)-1: words += ['*']
            counter += 1
        words += ['product equals'] 
        words += [str(self.question)]

        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset):
        result = 1
        for select in self.objs:
            subset = select(objset)
            if subset == 'INVALID':
                return 'INVALID'

            assert(len(subset)==1) 
            result *= subset[0].loc.value[0] * subset[0].loc.value[1]
        
        statement = True if self.question==result else False
        return statement

    def operator_length(self):
        return 2 + len(self.objs) # exist requires select + exist 

        
    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = []
            for loc in self.locs:
                should_be.append([stim.Object([loc])])
        else:
            result = self.question
            while result==self.question:
                result = 1
                should_be = []
                for _ in self.locs:
                    loc = stim.random_loc()
                    result *= loc.value[0] * loc.value[1]
                    should_be.append([stim.Object([loc])])

        if len(self.locs)==1:
            should_be = should_be[0]

        return should_be




#class Go_old(Get):
#    """Go to location of object"""
#    def __init__(self, objs):
#        super(Go, self).__init__('loc', objs)
#
#    def __str__(self):
#        return ' '.join(['point', str(self.objs)])
#
#    def operator_length(self):
#        return 1 # just a single operation for get

