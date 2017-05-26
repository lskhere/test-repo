# Submission to Challenge Task 1: Improve accuracy with GridSearch
 
 
## Challenge
* Demonstrate an understanding of hyperparameter optimization using sklearn GridSearch on a convolutional deep net against a simplified MNIST digit recognition by improving out-of-sample accuracy above 0.98398.
 
 
## Approach
* I first implement GridSearch using coarse grids on individual hyperparameters while holding all other parameters same as the base model. This gives me a rough idea which hyperparameters are more influential to our problem and allow me to narrow down to a finer set of hyperparameters for GridSearch in next step. 

* Next, I conduct GridSearch on combinations of selected hyperparameters. The best parameter set gives text accuracy 0.99xxx. 

* See *mnist-suki.ipyn* for the implementation algorithm.
 
 
## Improvement on Accuracy
* The following combination of parameters gives the best result on test accuracy,   .
 
 | Hyperparameter                 | Value                                                | Remarks                 |
 | ------------------------------ |----------------------------------------------------- |------------------------:|
 | no. of neurons                 | 32 (1st conv), 64 (2nd conv), 128 (fully connecetd)  | 
 | convolutional filter size      | 5x5                                                  |
 | batch size                     | 64                                                   |
 | no. of epochs                  | 128                                                  |
 | weight initialization          | unifrom                                              |
 | activation function            | ReLu                                                 |
 | optimizer                      | Adam                                                 |
 | dropout                        | 0.25 (fully-connected), 0.5 (output layer)           |

 
## Takeaway
* Compared with other ML algorithm, GridSearch can go crazy for neural net as there are numerous parameters when tuning a deep neural net and it is slow to train a deep neural net.
* To implement GridSearch, it is more efficient to start with coarse grids of individual hyperparameters on a smaller dataset, followed by delicate grids of combinations of selected hyperparameters. 
* It is important to understand the problem domain and leverage on previously applied techniques on the same (or similar) data problem. In our case, we start with a good enough CNN model on MNIST data and it is easier in hyperparameter tuning.
 
 
## Further Thoughts
* My blog post on [Tuning Hyperparameter for Deep Neural Nets](URL)
 
 
## Resources  
 * [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
 * [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3)
 * [Practical recommendations for gradient-based training of deep architectures by Yoshua Bengio](https://arxiv.org/abs/1206.5533)
 * [Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville](http://www.deeplearningbook.org/)
 
 
 
