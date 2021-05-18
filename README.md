# Grape-Category-Prediction
Classification of grape varieties in Python using Multinomial Logistic Regression, Decision Trees and Random Forests

Design a machine learning solution for determining the category or variety of grape used in making wines based on several chemical characteristics of individual wines. The attached dataset in csv format is the result of chemical analysis of wines grown in the same region in Italy but derived from three different cultivars (variety of grapes). This chemical analysis determined the quantities of 13 constituents found in each of the three types of wines. These 13 chemical characteristics are listed below and they impact the quality of wine in terms of taste, mouth fill, color, etc. 

1.	Cultivar – category/species of grape	8. 	Flavanoids
2.	Alcohol		9. 	NonFlavanoids
3.	MalicAcid		 10.  	Pcyamins
4.	Ash. 		11.	ColorIntensity
5.	Alkalinity (of ash)		12. 	Hue
6.	Magnesium 		13.	OD280
7.	Phenols 		14.	Proline

# Steps followed and questions answered,

1.	What exactly is the business problem you are trying to solve?  Summarize this in the form of a meaningful problem statement? 
2.	What are some of preliminary decisions you may need to make based on your problem statement? Your answer should include identification of an initial machine learning algorithm you will apply with respect to the problem statement in (1). Justification should be based on identification of the category of machine learning (supervised, unsupervised, etc.) as well as suggested machine learning algorithm from within the identified machine learning category. 
3.	Keeping your preliminary decisions from (2) in mind, peruse the dataset to:
a.	Display the datatype of each of the 14 columns to determine if any of the columns need to be transformed to comply with the requirements of your chosen algorithm. Specify the names of columns that require transformation along with the transformation that need to be performed. Include a reasonable explanation as to why the columns need to be transformed as well as what appropriate transformation will be necessary to make the feature algorithm-compliant.
b.	Identify any other data cleanup and pre-processing that may be required to get the data ready for your chosen machine learning algorithm. This may include handling missing values. Missing values for any feature are to be replaced with a median value for that feature. State so if missing values are not indicated. 
4.	Perform preliminary exploratory data analysis (EDA) pertinent to the problem statement and your chosen machine learning algorithm in (2). This may include basic statistics, data shape, grouping on the outcome variable, generating scatter plots or line plots, etc. as appropriate based on your chosen algorithm. Anything that can give you further insight into your dataset vis-à-vis the machine learning algorithm you have selected should be included with an explanation/conclusion of the output. 
5.	If your chosen algorithm demands training and test datasets, split your wine dataset using an 80/20 split. If dataset is split, evaluate your training and test datasets to ensure they are representative of your full data set. 
6.	Use the relevant portion of your dataset to train the model of your selected machine learning algorithm. Do all the necessary preprocessing to determine the parameters for your selected algorithm. For example, you will need to specify (and justify) the number of clusters if you choose to use KMeans clustering algorithm via the Elbow curve, Silhouette analysis, etc. 
7.	Using appropriate metrics for your chosen algorithm, evaluate the trained model. Explain and justify the worthiness of your trained model. 
8.	Next, use the relevant portion of your dataset (as dictated by the chosen algorithm) to evaluate the performance of your model. Again, use all relevant metrics for your algorithm to discuss the outcome in terms of model’s accuracy and usefulness in generating predictions. These may include such metrics as SSE, MSSE, Silhouette scores, completeness scores, confusion matrix, AOC curve, etc. as dictated by and available for your chosen machine language algorithm. 


