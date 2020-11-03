Project 2, NNs and SGD
---
>	work in progress

#Structure
The code lies in the `src` folder, with a `main.py` script which 
calls the systems, e.g. linear regression with the Franke function
which lie in the `systems` folder, with the named systems within. 
Methods for handling the mathematical models and the like lie in 
the library folder, `lib`. Here, for instance, is the `gradientdescent.py`
script containing the `SGD` class which performs stochastic gradient
descent on the features given to find an optimal set of coefficients for 
linear regression.
