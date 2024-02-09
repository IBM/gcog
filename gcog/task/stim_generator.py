import numpy as np
from collections import defaultdict
import random
import cv2
from bisect import bisect_left
import gcog.task.config as config

#### Define constants

class Attribute(object):
    """Base class for attributes of task features"""

    def __init__(self, value):
        self.value = value if not isinstance(value, list) else tuple(value)
        self.parent = list()

    def __call__(self, *args):
        """Including a call function to be consistent with Operator class."""
        return self

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        """Override the default Equals behavior."""
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __ne__(self, other):
        """Define a non-equality test."""
        if isinstance(other, self.__class__):
          return not self.__eq__(other)
        return True

    def __hash__(self):
        """Override the default hash behavior."""
        return hash(tuple(sorted(self.__dict__.items())))

    def resample(self):
        raise NotImplementedError('Abstract method.')

    @property
    def has_value(self):
        return self.value is not None

class Shape(Attribute):
    def __init__(self, value):
        super(Shape, self).__init__(value)
        self.attr_type = 'shape'

    def sample(self):
        # if want to randomly get another shape
        self.value = random_shape().value

    def resample(self):
        # pick another shape
        self.value = another_shape(self).value

class Color(Attribute):
    """Color class."""
    
    def __init__(self, value):
        super(Color, self).__init__(value)
        self.attr_type = 'color'

    def sample(self):
        # if want to randomly get another shape
        self.value = random_color().value

    def resample(self):
        # pick another shape
        self.value = another_color(self).value

class Loc(Attribute):
    """Location class."""

    def __init__(self, value=None):
        """Initialize location.

        Args: 
            value: None or a tuple of floats
            space: None or a tuple of tuple of floats
            If tuple of floats, then it's the actual location
        """ 
        super(Loc, self).__init__(value)
        self.attr_type = 'loc'

class Space(Attribute):
    """Space class -- empty cells

    Args:
        value: None or a tuple of floats
    """

    def __init__(self, value):
        super(Space, self).__init__(value)
        if self.value is None:
            #self._value = [(0, 1), (0, 1)]
            self._value = [0, 1]
        else:
            self._value = value

    def sample(self, avoid=None):
        """
        Samples a location.

        This function will search for a location to place the object that doesn't overlap with other objects at locations to avoi
        Places an object anyway if it couldn't find a good place
        """
        if avoid is None:
            avoid = []

        n_max_try = 1000
        for i_try in range(n_max_try):
            loc_x = np.random.randint(0,config.GRIDSIZE_X)
            loc_y = np.random.randint(0,config.GRIDSIZE_Y)

            overlapping = False
            for loc_avoid in avoid:
                overlapping_tmp = loc_avoid[0]==loc_x and loc_avoid[1]==loc_y
                # If overlaps with existing object, set as overlapping
                if overlapping_tmp: overlapping = True

            if overlapping==False:
                break

            if i_try==999:
                raise Exception("could not find a good location for correct stimulus")

        return Loc((loc_x, loc_y))

    def include(self, loc):
        """Check if an unsampled location (a space) includes a loc."""
        x, y = loc.value
        return (self._value[0] == x) and (self._value[1] == y)

class Object(object):
    """
    An object within the object set.
    An object is a collection of attributes.

    Args:
        loc: tuple(x,y)
        color: string ('red', 'green', 'blue', 'yellow', 'purple', 'orange', 'white')
        shape: string (a-z)
        when: (current, t-1, t-2) -- #TODO
        distractor: boolean. Whether or not this object is a distractor (and can be removed)

    Raises: 
        TypeError if loc, color, shape are neither None nor respective Attributes
    """

    def __init__(self, attrs=None, when=None, distractor=False):

        self.loc = Loc(None)
        self.color = Color(None)
        self.shape = Shape(None)
        self.space = Space(None)

        if attrs is not None:
            for a in attrs:
                if isinstance(a, Loc):
                    self.loc = a
                elif isinstance(a, Color):
                    self.color = a
                elif isinstance(a, Shape):
                    self.shape = a
                elif isinstance(a, Space):
                    self.space = a
                else:
                    raise TypeError("Unknown type for attribute: " + str(a) + ' ' + str(type(a)))

        self.when = when 
        self.distractor = distractor
        self.screen = None

    def __str__(self):
        return ' '.join([
            'Object:', 'loc',
            str(self.loc), 'color',
            str(self.color), 'shape',
            str(self.shape), 'when',
            str(self.when), 'screen',
            str(self.space), 'space',
            str(self.screen), 'distractor',
            str(self.distractor)
        ])

    def merge(self, obj):
        """Attempt to merge with another object.

        Args:
            obj: an Object instance

        Returns:
            bool: True if successfully merged, False otherwise
        """
        new_attr = dict()
        for attr_type in ['color', 'shape']:
            if not getattr(self, attr_type).has_value:
                new_attr[attr_type] = getattr(obj, attr_type)
            elif not getattr(obj, attr_type).has_value:
                new_attr[attr_type] = getattr(self, attr_type)
            else:
                return False
        
        for attr_type in ['color', 'shape']:
            setattr(self, attr_type, new_attr[attr_type])


    def create_img(self, width=64, height=64):
        """
        visualize the object
        """
        img_array = np.zeros((width, height,3))
        # pixel increments per grid element
        img = np.zeros((width,height,3),dtype=np.int16)
        color_str = self.color.value
        shape_str = self.shape.value
        x_pix = int(width/6) # approximately center within grid
        y_pix = int(height/1.25) # approximately center within grid
        #print(x_pix, y_pix)
        img = cv2.putText(img=img,text=shape_str,org=(x_pix, y_pix),
                          fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=4,color=config.WORD2COLOR[color_str],thickness=2)
        img_array[:,:,:] = np.squeeze(img).copy()
        img_array = np.asarray(img_array,dtype=int)

        return img_array

class ObjectSet(object):
#    """A collection of objects."""
    def __init__(self, whens):
        """
        Initialize the collection of objects

        """
        self.whens = whens
        self.set = list()
        self.end_screen = list()
        self.dict = defaultdict(list) # key: screen, value: list of object
        self.last_added_obj = None

    def __iter__(self):
        return self.set.__iter__()

    def add(self,
            obj,
            add_if_exist=False,
            delete_if_can=True):
        """Add an object.

        This function will attempt to add the obj if possible.
        It will not only add the object to the objset, but also instantiate the
        attributes such as color, shape, and loc if not already instantiated.

        Args:
          obj: an Object instance
          add_if_exist: if True, add object anyway. If False, do not add object if
            already exist
          delete_if_can: Boolean. If True, will delete object if it conflicts with
            current object to be added. Should be set to True for most situations.

        Returns:
          obj: the added object if object added. The existing object if not added.

        Raises:
          ValueError: if can't find place to put stimuli
        """

        if obj is None:
            return None
        
        # Check if object already exists
        obj_subset = self.select(space=obj.space,
                                 color=obj.color,
                                 shape=obj.shape,
                                 when=obj.when,
                                 delete_if_can=delete_if_can)

        if obj_subset and not add_if_exist: # True if more than zero satisfies
            self.last_added_obj = obj_subset[-1]
            return self.last_added_obj

        # Interpret the object
        if not obj.loc.has_value:
            # Randomly generate locations, but avoid objects that are already placed
            avoid = [o.loc.value for o in self.select(obj.when)]
            obj.loc = obj.space.sample(avoid=avoid)

        # if object has location specified, 
        # remove the previous object in its place (as long as it's a distractor)
        if obj.loc.has_value:
            for o in self.dict[obj.when]:
                if o.loc.value == obj.loc.value:
                    if o.distractor:
                        self.delete(o) # delete previous distractor in the same location
                    # if the object we're trying to replace is already there, then this is okay
                    if (o.loc.value==obj.loc.value) and (o.color.value==obj.color.value) and (o.shape.value==obj.shape.value):
                        continue
                    else:
                        # temporary hack
                        avoid = [o.loc.value for o in self.select(obj.when)]
                        obj.loc = obj.space.sample(avoid=avoid)
                        # get new location
                        #print('Trying to replace', o, 'with', obj)
                        #raise Exception("trying to place two non-distractor objects in same location; re-run")
                        
        if not obj.shape.has_value:
            obj.shape.sample()

        if not obj.color.has_value:
            obj.color.sample()

        #if obj.when is None:
        #    # if when is None, then object is always presented
        #    obj.screen = [0, self.n_screen]
        #elif obj.when =='current':
        #    obj.screen = [screen_now, screen_now + 1]
        #elif (obj.when == 't-1'):
        #    obj.screen = [screen_now-1, screen_now]
        #elif (obj.when == 't-2'):
        #    obj.screen = [screen_now-2, screen_now-1]

        ## Insert and maintain order
        #i = bisect_left(self.end_screen, obj.screen[1])
        #self.set.insert(i,obj)
        #self.end_screen.insert(i, obj.screen[1])

        ## Add to dict
        #for screen in range(obj.screen[0], obj.screen[1]):
        #    self.dict[screen].append(obj)
        self.dict[obj.when].append(obj)

        self.last_added_obj = obj
        return self.last_added_obj

    def add_distractor(self, when, available_distractors_colors=None, available_distractors_shapes=None):
        """Add a distractor."""
        if available_distractors_colors is not None and available_distractors_shapes is not None:
            color = random.choice(available_distractors_colors[when])
            shape = random.choice(available_distractors_shapes[when])
            attr1 = [Color(color), Shape(shape)]
        else:
            attr1 = random_colorshape()
        obj1 = Object(attr1, when=when, distractor=True)
        self.add(obj1, add_if_exist=True)

    def delete(self, obj):
        """Delete an object."""
        when = obj.when
        self.dict[when].remove(obj)

    def select(self,
               when,
               space=None,
               color=None,
               shape=None,
               delete_if_can=False):
        """
        Select an object satisfying properties.

        Args:
            when: The screen to select objects from
            space: None, or a Loc isntance, the loc to be selected
            color: None or a Color instance, the color to be selected
            shape: None, or a Shape instance, the shape to be selected
            delete_if_can: boolean, delete object found if can

        Returns:    
            a list of Object instances that fit the apttern provided by arguments
        """

        space = space or Space(None)
        color = color or Color(None)
        shape = shape or Shape(None)

        # select only objects that have happened
        subset = self.dict[when]

        if color is not None and color.has_value:
            subset = [o for o in subset if o.color == color]
        
        if shape is not None and shape.has_value:
            subset = [o for o in subset if o.shape == shape]

        if space is not None and space.has_value:
            subset = [o for o in subset if space.include(o.loc)]

        if delete_if_can:
            for o in subset:
                if o.distractor:
                    # delete obj from self.
#                    print(o)
                    self.delete(o)
            # keep the not-deleted
            subset = [o for o in subset if not o.distractor]

        # order objects by location to have deterministic ordering
        subset.sort(key=lambda o: (o.loc.value, o.color.value, o.shape.value))

        return subset

    def create_img(self, width=200, height=200):
        """
        make the image readable
        """
        img_array = np.zeros((width, height, 3, len(self.whens)))
        # pixel increments per grid element
        dx = int(width/config.GRIDSIZE_X)
        dy = int(height/config.GRIDSIZE_Y)
        e_count = 0
        for screen in self.whens:
            img = np.zeros((width,height,3),dtype=np.int16)
            for o in self.dict[screen]:
                x, y = o.loc.value
                color_str = o.color.value
                shape_str = o.shape.value
                x_pix = int(x*dx) + int(dx/3) # approximately center within grid
                y_pix = int(y*dy) + int(dy/1.5) # approximately center within grid
                #print(x_pix, y_pix)
                img = cv2.putText(img=img,text=shape_str,org=(x_pix, y_pix),
                                  fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=config.WORD2COLOR[color_str],thickness=1)
            img_array[:,:,:,e_count] = np.squeeze(img).copy()
            e_count += 1

        return img_array

    def objset2vecID(self):
        """
        converts objset to matrix form
        """
        objset_mat = np.zeros((config.GRIDSIZE_Y,
                               config.GRIDSIZE_X,
                               2, # two feature dimensions per element (color + shape)
                               len(config.ALL_TIME)))
        t = len(config.ALL_TIME)-1 # iterate backwords in time
        for time in config.ALL_TIME:
            for obj in self.dict[time]:
                y, x = obj.loc.value
                color_idx = config.ALL_COLORS.index(obj.color.value) + 1
                shape_idx = config.ALL_SHAPES.index(obj.shape.value) + 1
                #vec_colorshape = np.asarray(self._colorshape2vec(obj.color, obj.shape),dtype=bool)
                objset_mat[y,x,0,t] = color_idx
                objset_mat[y,x,1,t] = shape_idx
            t -= 1

        return objset_mat

    def _colorshape2vec(self,color,shape):
        """
        returns a vector encoding the color and shape
        returns a 1d vector of length num_colors + num_shapes 
        the first num_colors elements correspond to the one-hot color encoding 
        the second num_shapes elements correspond to the one-hot shape encoding
        together, the 1d array will sum to 2.
        """
        num_colors = len(config.ALL_COLORS)
        num_shapes = len(config.ALL_SHAPES)
        color_str = color.value
        color_ind = config.ALL_COLORS.index(color_str)
        shape_str = shape.value
        shape_ind = config.ALL_SHAPES.index(shape_str)
        vec = np.zeros((num_colors+num_shapes,))
        vec[color_ind] = 1
        vec[num_colors+shape_ind] = 1
        return vec

class StaticObject(object):
    """Object that can be loaded from a dataset and rendered."""
    def __init__(self, loc, color, shape, screen):
        self.loc = loc
        self.color = color
        self.shape = shape
        self.screen = screen

def vec2objset(vec):
    """
    Given the 3-dim matrix (space, colorshape, time)
    this function returns an ObjectSet instance corrsponding
    to the exact same vec
    """
    num_colors = len(config.ALL_COLORS)
    num_shapes = len(config.ALL_SHAPES)
    y_vec, x_vec, colorshape, time_vec = np.where(vec)
    objsetcopy = ObjectSet(whens=config.ALL_TIME)
    for y, x, t in zip(y_vec,x_vec,time_vec):
        color_ind = np.where(vec[y,x,:num_colors,t])[0][0]
        shape_ind = np.where(vec[y,x,num_colors:,t])[0][0]
        n_time = len(config.ALL_TIME)-1
        when = config.ALL_TIME[n_time-t]
        loc = Loc((y,x))
        shape = Shape(config.ALL_SHAPES[shape_ind])
        color = Color(config.ALL_COLORS[color_ind])
        obj = Object([loc,color,shape],when=when)
        objsetcopy.add(obj)
    return objsetcopy

def random_attr(attr_type):
    if attr_type == 'color':
        return random_color()
    elif attr_type == 'shape':
        return random_shape()
    elif attr_type == 'loc':
        #return Loc([round(random.uniform(0.05,0.95), 3),
        #            round(random.uniform(0.05, 0.95), 3)])
        return Loc([np.random.randint(0,config.GRIDSIZE_Y), np.random.randint(0, config.GRIDSIZE_X)])
    else:
        raise NotImplementedError('Unknown attr_type :', str(attr_type))

def random_shape():
    return Shape(random.choice(config.ALL_SHAPES))

def random_color():
    return Color(random.choice(config.ALL_COLORS))

def random_loc():
    return Loc((random.choice(range(config.GRIDSIZE_X)),random.choice(range(config.GRIDSIZE_Y))))

def another_shape(shape):
    all_shapes = list(config.ALL_SHAPES)
    try: 
        all_shapes.remove(shape.value)
    except AttributeError:
        for s in shape:
            all_shapes.remove(s.value)
    return Shape(random.choice(all_shapes))

def another_color(color):
    all_colors = list(config.ALL_COLORS)
    try: 
        all_colors.remove(color.value)
    except AttributeError:
        for c in color:
            all_colors.remove(c.value)
    return Color(random.choice(all_colors))

def another_attr(attr):
    if isinstance(attr, Color):
        return another_color(attr)
    elif isinstance(attr, Shape):
        return another_shape(attr)
    elif isinstance(attr,Space):
        raise NotImplementedError()
    elif attr == 'INVALID':
        return attr

def random_colorshape():
    return random_color(), random_shape()

def random_time():
    return np.random.choice(config.ALL_TIME)





