Thoughts on reading
---

I've """finished""" writing a report on the...progress of the 
project. It's a definitive draft and I need to reread, note and
edit it. 

notes
----
Should compare with guidelines in 
1. Course repo
2. Video on report writing
3. Example report
4. Report from last year
5. Consider subsections for explanative purposes

###Abstract
Odd wording throughout. Should probably rewrite from memory to
...chew on it a bit more. 
1.	"the project motivation lies..." <= it's off. 
2.	description of what is/should have been done needs redoing
3.	reword description of why project didn't finish
Exploration of machine learning methods by the Regression subfamily.
The methods were tested on a simulated surface generated by the Franke function.
Initially, regressed with Ordinary least squares regression before 
expanding to Ridge. Exploration of resampling methods also performed, with 
bootstrap and k-fold CV. Intent was to also implement the LASSO regression method
and introduce real data to the models. This fell through.
###Introduction
Still with the strange language. Need to rewrite this too. Both
intro and abstract are short enough that it would work. 
points
1.	Introduction and description of Franke's function. 
2.	Explain the steps of the process more coherently - some elaborations
	some...streamlining
3.	The...involvement of the real data needs revisiting. both based on the 
	project description and the general understanding
The motivation for this project was in large part to explore regression methods as a 
subset of machine learning. Machine learning bases itself mainly on a frequentist approach,
or frequency or proportion of data as a metric for prediction. The regression methods have
a relatively intuitive setup and serves as a good entry point for exploring machine learning
as well as being usefull for predicting numerical problems suspected of following an unknown
function. 
Three methods were decided on to explore, ordinary least squares(OLS), Ridge regression and 
Lasso regression. OLS was performed with results approximating the expectations for regressions
of our dataset. Ridge was barely runnable, but lacked any finish wrt. storing and displaying the 
data from it. Lasso regression fell off completely. 
Beyond the regressor models, we also utilized resampling methods, Bootstrap and k-fold Cross Validation.
These methods allow us to make the best use out of the data already in our possession, as the resampling
helps average over several smaller "data sets" sampled from a majority of the initial data. The results 
from these smaller sets are compared to the minority of the original which has been left untouched. The
amount of tests allows us to controll for e.g. variance and bias. 
Once these methods were laid out and tested, the idea was to move on to real data, rather than the simulation
and make use of terrain data from Norway. Due to time constraints, this did not happen. 
The report here is tructured into a method section where I generally go through the assumptions, models and 
algorithms for the methods mentioned. Following this, is a section of the few results I managed to collect 
and an attempt at discussing them as they relate. Finally, the conclusion attemtps to conclude what can be 
from concluded after the run. Also here, I attempt to describe the issues with the implementation as well
as the state of the project and how they might be improved.
###Methods
first, ad the s to the section. I think. Look, jus- just look it up.*waves hands*
1.	Possibly reshape franke function mathmode presentation? 
2.	Do i want to start cold with the franke function?
3.	rework description of Franke and system for the surface we set up
4.	No, I won't stop going between 'we' and 'I' completely at random. :3
5.	Wording on the setup of the..domain of the function x,y = [0,1] needs
	rewording
6.	The explanation of the regression..case/model/..assumption(?) needs rework
7.	The algorithm for making the feature matrix could need a once-over
	Including bit more in the caption. 
8.	Missing close ) in current eq.11
9.	Overal, justify more for everything in section and consider changing order of 
	presentation
10.	The description of resampling is...lacking
11.	Caption for the bootstrap algo needs to be rewritten with basis in theory for 
	bootstrap
12.	Justify the other regression methods better. => with basis in theory
13.	join in description of the addition to the MSE for the norms 1 and 2 wrt. regression
	rewrite in 12
14.	"the code is set up so that..." the word 'three' is misspelled.
15.	The explanation between align 19 and 20 requires rewrite. -> what does wolfram tell us?
###Results
1.	Start the section differently. Consult sample report and previously written one.
2.	Caption for the plot of the franke surface needs to be retouched.
3.	Figure references for MSE train/test and bias-variance is off. Likely referencing something in the
	appendix
4.	While there were very few results, they may benefit from greater explanation/introduction

5.	Should probably include/describe the input variables in the code for the results presented
###Conclusion
1.	Consult the references as to beginning the section
2.	Go into the representative results each. Both of..them...
3.	MIGHT FIT IN RESULTS TOO: consider retouching figures 2 and 3.
4.	The issues and possible improvements to the code should be retouched to possibly structure
	a little better and provide clarity. 
###Appendix
include subsections wrt. the plots to include/improve lookup

Notes on sample report
---
1.	Don't need table of contents
