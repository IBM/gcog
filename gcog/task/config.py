#### STIMULUS OPERATORS
ALL_SHAPES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'white','pink','cyan','brown'] #, 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white']
ALL_TIME = ['current'] # make sure current is twice as likely as any other option (only one stimulus image in this implementation)
ALL_TIME_PROB = [0.5, 0.25, 0.25] # make sure current is twice as likely as any other option (not used in current implementation)
WORD2COLOR = {
    'red': (230, 25, 75),
    'green': (60, 180, 75),
    'blue': (0, 130, 200),
    'yellow': (255, 225, 25),
    'purple': (145, 30, 180),
    'orange': (245, 130, 48),
    'cyan': (70, 240, 240),
    'magenta': (240, 50, 230),
    'lime': (210, 245, 60),
    'pink': (250, 190, 190),
    'teal': (0, 128, 128),
    'lavender': (230, 190, 255),
    'brown': (170, 110, 40),
    'beige': (255, 250, 200),
    'maroon': (128, 0, 0),
    'mint': (170, 255, 195),
    'olive': (128, 128, 0),
    'coral': (255, 215, 180),
    'navy': (0, 0, 128),
    'grey': (128, 128, 128),
    'white': (255, 255, 255)
}
GRIDSIZE_X = 10
GRIDSIZE_Y = 10
ALL_LOCATIONS = []
for x in range(GRIDSIZE_X):
    for y in range(GRIDSIZE_Y):
        ALL_LOCATIONS.append(tuple((y,x)))

#### OPERATOR CONSTANTS
# operators with multiple inputs -- not used paper
MULTI_OBJ_OPS = ['existand', 'existor', 'existxor', 
                 'sumeven', 'producteven','sumodd','productodd',
                 'samecolor','sameshape','notsameshape','notsamecolor'] # 'add', 'subtract', 'multiply',
# operators with single input
SINGLE_OBJ_OPS = ['exist', 'go', 'getcolor','getshape',
                  'sumeven','producteven','productodd','sumodd',
                  ] 
CONNECTOR_OPERATORS = ['switch'] # aka 'if-then-else'
ALL_TASK_OPS  = SINGLE_OBJ_OPS
# operators that can feed into a if-then-else operator (i.e., output is true/false)
STARTING_OPS = ['exist',
                'sumeven', 'producteven','sumodd','productodd',
                ] #'add', 'subtract', 'multiply',
# operators that tasks can end on
ENDING_OPS = ['exist',
              'sumeven', 'producteven','sumodd','productodd',
              'getcolor', 'getshape', 'go'] # 'add', 'subtract', 'multiply',

# specify operators that return specific colors
GETCOLOR = ['getcolor','iscolor','samecolor','notsamecolor']
# specify operators that return specific shapes
GETSHAPE = ['getshape','isshape','sameshape','notsameshape']

#### OUTPUT OPERATORS
OUTPUT_UNITS = [True] + [False] + ALL_LOCATIONS + ALL_SHAPES + ALL_COLORS

#### SPECIFY INPUT OPERATORS
SELECT_INPUT_ARRAY  = ALL_SHAPES + ALL_COLORS + ALL_TIME # don't specify location in the select input array
SELECT_ARRAY_LENGTH = 3 # shape/color/time
MAX_SELECT_OPS = 2 # maximum number of select operators per function/operator node
ALL_OPERATORS = ALL_TASK_OPS + CONNECTOR_OPERATORS
RULE_ARRAY_LENGTH = SELECT_ARRAY_LENGTH*MAX_SELECT_OPS + 1 + 1 # operator ID + EOS token at the end
STIMULI_ARRAY_LENGTH = 2 + 1 # 2 feature dimensions (color, shape) + EOS symbol
