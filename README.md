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

Each file can be executed with a ```-run ``` parameter where the accepted values are:
- training - runs given script against training dataset
- test - runs given script against test dataset

For example, to run ```python assignment_q1.py``` against test dataset one would use:

```python assignment_q1.py --run test```

and to run same script against training dataset one would use:

```python assignment_q1.py --run training```

Those operations are supported for all files. Furthermore if ```--run``` parameter is omitted then script selected will run against test dataset by default

### Additonal parameter for weighted kNN and kNN regression
For ```python assignment_q2.py``` as well as ```python assignment_q3.py``` there is an extra ```--run``` parameter value accepted called - ```best``` 
"Best" will cycle through values of K from 1 to 10 (inclusive) and through 3 different distance algorithms:
- Euclidean
- Manhattan
- Minkowski

for the test dataset and prints resulting accuracy values, per k, per distance algorithm, to the console.

Complete command would look like
```python assignment_q2.py --run best```
or
```python assignment_q3.py --run best```

### Additional parameters for weighted kNN
For the ```python assignment_q2.py``` alone there
#### ```--run mink```
"Mink" param when supplied to the run parameter will cycle through a number of k and p values and execute analysis on test data using Minkowski distance algorithm
```python assignment_q2.py --run mink```