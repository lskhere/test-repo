# Submission to Challenge Task 1: Improve accuracy with GridSearch
 
## Challenge
* Demonstrate an understanding of hyperparameter optimization using sklearn GridSearch on a convolutional deep net against a simplified MNIST digit recognition by improving out-of-sample accuracy above 0.98398.
 
## Overview of My Approach
I first implement coarse grid search using 3-fold CV on individual set of hyperparameters while holding all other parameters same as the base model. This gives me a rough idea which hyperparameters are more influential to our problem and allow me to narrow down to a finer set of hyperparameters for GridSearch in next step. 
Next, I conduct a finer GridSearch on combination of selected hyperparameters. The best set of parameters gives text accuracy 0.99xxx. See *mnist-suki.ipyn*.
 
## Improvement
* Accuracy of the best parameter set
 
| GridSearch on Hyperparameters                      | Best Parameter Set  |  Accuracy|
| ---------------------------------------------------------------- |------------------------:|
|   {"neurons" : [[4,8,16],[32,64,128]], 
   "batch_size":[64,128], 
   "epochs": [10, 20]] }                                                                 |  94.4.%                      |
| No. of neurons                                       |                                                         |              |
| Batch size and No. of epochs              | {“batch_size”: 64, “epochs”: 20}     | 0.99091 |
| Weight initialization                               | “he_uniform”                                    |0.98977|
| Activation function                                 | “relu”                                                | 0.98936|
| Optimizer                                               | “Adagrad”                                        | 0.99255|
| Dropout                                                 | xx                                                     | 
 
 

## Takeaway
* Compared with other ML algorithm, GridSearch can go crazy for neural net as there are numerous parameters when tuning a deep neural net and it is slow to train a deep neural net.
* To implement GridSearch, it is more efficient to start with coarse grids of individual hyperparameters on a smaller dataset, followed by delicate grids of combinations of selected hyperparameters. 
* It is important to understand the problem domain and leverage on previously applied techniques on the same (or similar) data problem. In our case, we start with a good enough CNN model on MNIST data and it makes life easier in hyperparameter tuning.
 
## Further Thoughts
My blog post on [“Tuning Hyperparameter for Deep Neural Nets” ](URL)
 
## Resources  
[Resource1](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[Resource2](http://cs231n.github.io/neural-networks-3)

[Resource3](https://arxiv.org/abs/1206.5533)

[Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville](URL)
 
 
 
