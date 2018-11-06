# Practical Machine Learning assignment 1

### Assignment files
The assignment is separated into 3 files, one for each part. 
- ```assignment_q1.py``` corresponds to basic implementation on k-nearest neighbours (kNN).
- ```assignment_q2.py``` corresponds to part 2, where kNN is weighed
- ```assignment_q3.py``` corresponds to the kNN regression problem

### Additional files

- ```common_utils.py``` holds functions that are shared across different parts of the assignment
- ```test_assignment.py``` holds unit tests on some of the supporte functionality that is critical to the success of the algorithm. Tests include ability to:
    - load data
    - calculate classification accuracy correctly
    - clean cancer dataset
    
# How to run
## Prerequisites
1. Assignment files are intended to run with Python version 3.4 and above.
2. Anaconda interpreter, version 3.x is recommended, otherwise the following libraries will need to be installed before files can be run
    - numpy
    - pandas 
 
## Running of files
Each ```assignment_x``` file should be run individually. Depending on target python setup the command to run will look similar, if not the same, to

```python assignment_q1.py```

```python assignment_q2.py```

```python assignment_q3.py```

No commandline arguments are supported at this time.