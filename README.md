# gCOG
*Generic COG (gCOG): A Compositional Generalization Dataset to Evaluate Multimodal Reasoning*

#### Contact: takuya.ito@ibm.com

**Reference:** Ito T, Dan S, Rigotti M, Kozloski J, Campbell M (2024). On the generalization capacity of neural networks during generic multimodal reasoning. International Conference on Learning Representations (ICLR). [http://arxiv.org/abs/2401.15030](http://arxiv.org/abs/2401.15030).

#### Last update: 2/12/2024

## Dependencies
Dependencies: `environment.yml` (`conda`) and `requirements.txt` (`pip`)

## Installation and setup
1) Clone the github repository
2) Set up the environment and dependencies, if needed (`conda env create -f environment.yml` and `pip -r install requirements.txt`)
3) Install package from base directory: `pip install -e .`

## Overview
**A task paradigm to systematically evaluate compositional generalization features.** 
The purpose of this task is to measure how models compositionally generalize over precise task features, such as:
1) Systematic compositional generalization: Task operations (e.g., re-using previously seen task operations in novel settings)
2) Productive compositional generalization: Task complexity (e.g., depth / complexity of a task tree).
3) Distractor generalization: Stimulus/noise distractors (inclusion of irrelevant task information in the stimulus image).

This task was originally derived from the COG dataset and its predecessors [(Yang et al. (2018))](https://arxiv.org/abs/1803.06092), but includes different task operators and specific dataloaders with explicit training/testing splits.
Dataset and splits are provided in two categories:
* Abstract/categorical tokens (as evaluated in the paper)
* Images in pixel and instructions in language representations (see `demos/PixelWordDataset/` for examples)
* The primary code associated with the task is housed in `gcog/task/`

### Demo notebooks with details on specific splits
There are 4 demo notebooks corresponding to each compositional split presented in the paper (distractor, systematicity (depth 1), systematicity (depth 3), productive generalization), as well as a generic notebook that demos the underlying task framework. The demos provide sample code for how to generate train/test splits.

**1) Task framework: `demos/Demo_TaskGeneration.ipynb`**

**2) Distractor generalization demo: `demos/Demo_DistractorGeneralization_Fig3split.ipynb`**
<img width="577" alt="image" src="https://github.com/IBM/gcog/assets/6352881/8cd7c2b8-80a3-47ca-a607-12e0e5146451">

**3) Systematic generalization (task tree depth 1) demo: `demos/Demo_Systematicity_OpSys_Fig4Asplit.ipynb`**
<img width="202" alt="image" src="https://github.com/IBM/gcog/assets/6352881/d72b25cd-0c06-4b94-9a62-ada01f4d7629">

**4) Systematic generalization (task tree depth 3) demo: `demos/Demo_Systematicity_CompTreeSubsets_Fig4Dsplit.ipynb`**
<img width="202" alt="image" src="https://github.com/IBM/gcog/assets/6352881/18c8beb7-e461-4a51-bf24-fa40bfe04bbf">

**5) Productive generalization demo: `demos/Demo_Productivity_CompTree_Fig5split.ipynb`**
<img width="615" alt="image" src="https://github.com/IBM/gcog/assets/6352881/d671104e-c909-47b6-b741-c7e54d389bdb">

In addition, there are a separate set of demo notebooks that provide instructions for how to import and/use dataloaders that generate task samples in the image pixel and language instructions (i.e., not categorical tokens): `demos/PixelWordDataset/`

## Description of task operators

**Exist:** Asks if a specific object exists. Specified with a color and shape (letter). 
* Example: "*Is there a 'red c'?*" 
* Returns: True or False.

**GetColor/GetShape:** Asks agent to return either a color or shape of an object with a specified attribute. 
* Example: "*Get the shape of the green object*" 
* Example: "*Get the color of the letter a*". 
* Returns: A string attribute (e.g., "a" or "red").
* N.B.: There are checks in the task program to ensure that this question is not ill-posed (i.e., only a single correct answer).

**Go:** Asks the agent to return the location of a specified object.
* Example: "*Get the location of the 'red c'*"
* Returns: A tuple (x, y) coordinates.

**AddEven:** Asks the agent to add the location values of an object(s), and asks if the sum is even.
* Example: "*Is the sum of the coordinate values of the 'red c' even?*"
* Answer: If the 'red c' is on coordinate (4, 6), then the correct answer is True (4 + 6 = 10 is even)
* Example: "*Is the sum of the coordinate values of the 'red c' and the 'blue k' even?*"
* Answer: If the 'red c' is on coordinate (4, 6), and the 'blue k' is on (1, 4), then the correct answer is False (4 + 6 + 1 + 4 = 15 is odd)

**AddOdd:** Asks the agent to add the location values of an object(s), and asks if the sum is odd.
* Example: "*Is the sum of the coordinate values of the 'red c' odd?*"
* Answer: If the 'red c' is on coordinate (4, 6), then the correct answer is False (4 + 6 = 10 is even)
* Example: "*Is the sum of the coordinate values of the 'red c' and the 'blue k' odd?*"
* Answer: If the 'red c' is on coordinate (4, 6), and the 'blue k' is on (1, 4), then the correct answer is True (4 + 6 + 1 + 4 = 15 is odd)

<!-- **Subtract:** Asks the agent to subtract the location values of an object(s)
* Example: "*Subtract the coordinate values of the 'red c' on the current window*"
* Answer: If the 'red c' is on coordinate (4, 6), then the correct answer is - 4 - 6 = -10
* Example: "*Add the coordinate values of the 'red c' and the 'blue k' on the current window*"
* Answer: If the 'red c' is on coordinate (4, 6), and the 'blue k' is on (1, 3), then the correct answer is - 4 - 6 - 1 - 3 = -14 -->

**MultiplyEven:** Asks the agent to multiply the location values of an object(s), and asks if the product is even.
* Example: "*Is the product of the coordinate values of the 'red c' even?*"
* Answer: If the 'red c' is on coordinate (4, 6), then the correct answer is True (4 * 6 = 24 is even)
* Example: "*Is the product of the coordinate values of the 'red c' and the 'blue k' even*"
* Answer: If the 'red c' is on coordinate (4, 6), and the 'blue k' is on (1, 3), then the correct answer is True (4 * 6 * 1 * 4 = 96 is even)

**MultiplyOdd:** Asks the agent to multiply the location values of an object(s), and asks if the product is odd.
* Example: "*Is the product of the coordinate values of the 'red c' odd?*"
* Answer: If the 'red c' is on coordinate (4, 6), then the correct answer is False (4 * 6 = 24 is even)
* Example: "*Is the product of the coordinate values of the 'red c' and the 'blue k' odd*"
* Answer: If the 'red c' is on coordinate (4, 6), and the 'blue k' is on (1, 3), then the correct answer is False (4 * 6 * 1 * 4 = 96 is even)

<!--
## Multi-input operators (only multiple inputs accepted):

**ExistAnd:** Asks if a multiple objects exist (>2 inputs accepted). Specified with a time ('current', 't-1', or 't-2'), color, and shape (letter). 
* Example: "*Is there a 'red c' AND 'blue k' on the current window?*" 
* Example: "*Was there a 'red c' AND 'blue k' AND 'blue l' two screens ago?*" 
* Returns: True or False.

**ExistOr:** Asks if one or more objects exist (>2 inputs accepted). Specified with a time ('current', 't-1', or 't-2'), color, and shape (letter). 
* Example: "*Is there a 'red c' OR 'blue k' on the current window?*" 
* Example: "*Was there a 'red c' OR 'blue k' OR 'blue l' two screens ago?*" 
* Returns: True or False.

**ExistXor:** Asks if exclusively one objects exists, given multiple object inputs (>2 inputs accepted). Specified with a time ('current', 't-1', or 't-2'), color, and shape (letter). 
* Example: "*Is there a 'red c' XOR 'blue k' on the current window?*" 
* Example: "*Was there a 'red c' XOR 'blue k' OR 'blue l' two screens ago?*" 
* Returns: True or False.

**IsColor/IsShape:** Asks if the attribute of an object is correct. Specified with a time ('current', 't-1', or 't-2').
* Example: "*Is the color of the current 'a' red?
* Example: "*Is the shape of the current red object an 'a'?

**IsSameShape/IsSameColor:** Asks if the attributes of two or more objects are the same (>2 inputs required). Specified with a time ('current', 't-1', or 't-2'), and then color OR shape (letter). 
* Example: "*Is the color of the 'c' and the color of the 'k' the same on the current window?*" 
* Example: "*Is the shape of the red object and the shape of the blue object the same one window ago?*" 
* Returns: True or False.

**NotSameColor/NotSameShape:** Asks if the attributes of two or more objects are all different (>2 inputs required). Specified with a time ('current', 't-1', or 't-2'), and then color OR shape (letter). 
* Example: "*Is the color of the 'c' and the color of the 'k' different on the current window?*" 
* Example: "*Is the shape of the red object and the shape of the blue object different two windows ago?*" 
* Returns: True or False.
* -->

**If-Else** This is a connector operator, which connects two subtrees together by inputting a boolean, and then branching into two different directions. If the boolean is True (e.g., the output of an Exist operator), then the agent should follow the left branch. If False, the agent should follow the right branch. This connector operator enables queries to be arbitrarily long.



