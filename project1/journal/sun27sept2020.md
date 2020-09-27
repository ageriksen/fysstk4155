
Feature matrix function
-----------------------

The main point is that each implementation of the actual collumn is
to go through each "x_i", so that we multiply any permutation of the 
different x's to a maximum power of the degree of the polynomial we want
to create. e.g. x, y for a 3rd degree poly, 

x^3 + x^2y + xy^2 + y^3

and so on. 

###Collumn designation
While this is fair enough, another issue is the indexing of the collumn you want to 
store it to. The function I found has a pretty implementation of a multiplication factor 
to set the collumn, though I suspect it can also be done through a simple "k++" part in the
loop. 

###Designating amount of collumns
This is the real issue at this point. I know it's a mathematical relation, and I believe
there is a thread on stackoverflow/mathoverflow linked through piazza course page. 
At this point I don't think I want to go too in depth on how this goes, to focus more on making the
machine learning work. 


Train and test split of data
----------------------------
I changed the point of the "split_data" function to return
the indices rather than the new arrays. Splitting the data based 
on the raveled "row_arr" posed an issue with out of bounds indexing, but
splitting based on the feature matrix allows both splitting the z-array and 
the "row_mat" into proper splits. At least seemingly.

Noise on the Franke function
----------------------------
There's several things to consider regarding the noise of the franke function.
First, discussions on Piazza, Morten stated in questions "@29" 
> ...adding this in the x and y directions is what makes most sense,...
I also have the benefit of a previous iteration of the project. Here, the input data to the 
franke function is made with 
```python
nrow = 100
ncol = 200
rand_row        =       np.random.uniform(0, 1, size=nrow)
rand_col        =       np.random.uniform(0, 1, size=ncol)

sortrowindex    =       np.argsort(rand_row)
sortcolindex    =       np.argsort(rand_col)

rowsort         =       rand_row[sortrowindex]
colsort         =       rand_col[sortcolindex]
```
which is in line with Morten's one-dim example 
```python
x = np.random.rand(100,1)
fx = 5*x+0.01*np.random.randn(100,1)
```
And further in my previous code, a 2-dim noise element
```python
noiseSTR        =       0.1
noise           =       np.random.randn(nrow, ncol)

zmat_nonoise    =       Frankefunction(rowmat, colmat)
zmat            =       zmat_nonoise + noiseSTR*noise
```
