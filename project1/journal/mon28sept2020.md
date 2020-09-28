Singular matrix (X^T@X)^-1 
--------------------------

the files are similar, one is the original, while the other is an attempt 
at making some order of the first chaos as well as accessing the indices 
for train and test in order to plot the predicted terrain
###Solution(?)
I think the issue was in the creation of the feature matrix. I used the vanilla
x and y arrays, before meshgrid and ravel to create the X matrix. fixing this seems to have
solved the issue.
Definitely seems like it!
	1. The diff function proved the correct use here. 

Plotting the terrain prediction
-------------------------------
Another issue with plotting the predictions, is that the input arrays have been flattened from their
previous matrix state, so the dimensions of the prediction and the input is different. 
	1. Could one use the flattened arrays to plot the surface? 
	2. Could one "unflatten" the prediction?
		*. Them could one reliably map it back onto the mesh?

Adding scaling 
-----------
The requirement to scale the data was accomplished with the `np.mean()` method, s.a. 
```python
X_scaled = X-np.mean(X)
```
This is mainly from a discussion on piazza where the sklearn `StandardScaler`
had some strange behaviour. Might replace later. might possibly make a function
or a lambda function for it. 

