#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # <font color='black'>Question1.What exactly is the business problem you are trying to solve? Summarize this in the form of a meaningful problem statement? </font>

# ## <font color='purple'>Determining the category or variety of grape used in making wines based on several chemical characteristics of individual wines</font>

# # <font color='orange'> Problem Statement</font>

# ## <font color='purple'>Predicting the category or variety of grape used in making wines based on the chemical composition of 13 constituents found in each of the three types of grape</font>
# 

# In[ ]:


# Data Required (We need to have data related to charateristics based on which a particular class is assigned to train our model)


# # Question2. What are some of preliminary decisions you may need to make based on your problem statement? Your answer should include identification of an initial machine learning algorithm you will apply with respect to the problem statement in (1). Justification should be based on identification of the category of machine learning (supervised, unsupervised, etc.) as well as suggested machine learning algorithm from within the identified machine learning category. 

# ## <font color='purple'>The business problem we are trying to solve is predicting the type of wine where we are dealing with labeled data.(Cultivar) </font>
# 

# # <font color='orange'> Identification of the category of Machine Learning</font>

# ## <font color='purple'>As we have a classified data where there is a target class(Cultivar) and some characteristics based on which this target is classified, we need to train the model to perform same process </font>

# ## <font color='purple'>The category of machine learning which deals with labeled data is Supervised Learning </font>
# 

# ## <font color='purple'>In Supervised learning both input features  and target variable are available for each training data. A supervised learning algorithm then tries to learn the relationship between the input and output variables from the data, so that when input x is given, it is able to produce the corresponding output y. And to do so, the algorithm iterates the training data to adjust the parameters of a model, until it could generalize the data. This is called the learning process. </font>
# 

# ## <font color='purple'>As we are trying to determine(predict) the category of grape which is a labeled variable we need to choose between the sub division of Supervised Learning  </font>
# ## <font color='orange'>Regression and Classification</font>
# ## <font color='purple'>Regression is used when the target variable is numeric and continuous </font>
# ## <font color='purple'>Classification is used when the target variable is categorical </font>
# ## <font color='purple'>Our problem has target variable as categorical type (Type of grape) </font>

# ## <font color='purple'>So under the types of supervised learning algorithms we would eliminate the choice of Linear Regression as it deals with continuous variables  </font>

# ## <font color='purple'>We need to choose from the further available alogorithms Logistic Regression, Decision Tree and Random Forest</font>

# # <font color='orange'> Identification of the category of Machine Learning Algorithm </font>

# ## <font color='purple'>While all three of the remaining algorithms are applicable to achieve the objective, we will start with Logistic Regression.</font>

# ## <font color='purple'>The time consumed and complexity of logistic regression model is less when compared to Decision Trees and Random Forests. As there will be creation a multiple branches or if else loops created in case of Decision trees and random forests resulting more time consumption for training the model </font>

# ## <font color='purple'>As there are three different categories in the target variable so we choose Multinomial Logistic Regression which classifies based on one versus rest method</font>

# ### In Multi-class logistic regression creates different groups using one versus ret methodology, 
# ### For example Cultivar1 class the outputs are considered as 1,-1,-1 for Cultivar1, Cultivar2, Cultivar3 respectively  
# ### In case of Cultivar2 class the outputs are considered as -1,1,-1 for Cultivar1, Cultivar2, Cultivar3 respectively
# ### Similarly for Cultivar3 is as -1,-1,1 for Cultivar1, Cultivar2, Cultivar3 respectively
# ### So after this the first model is created as M1 based on input features and the first column outputs for example from above conditions 1,-1,-1 and 
# ### this model(M1) will be able to predict if the output is Cultivar1 or not
# ### Similarly M2 model  will be created for Cultivar2 as output
# ### Similarly M3 model  will be created for Cultivar3 as output

# ### when test data is given then the output would be calculated from three model M1, M2, M3 (probabilities as otputs)
# ### Were sum of the three  probabilites is equal to 1
# ### For generation og prediction the array of three probabilities is considered and the one with highest probability is considered to be the prediction output
# ### So if the output probabilities are [0.25, 0.25, 0.5] the output would be Cultivar3

# In[2]:


#Loading the pandas library
import pandas as pd


# In[3]:


#Loading the numpy library
import numpy as np


# In[4]:


#Loading sklearn for machine learning packages
import sklearn


# In[5]:


## importing the wine dataset with pandas
Wine_DF = pd.read_csv('D://wine.csv', header=0, sep=',')


# In[6]:


#Seeing the shape of the dataset
print("Shape of the data contained in wine.csv is", Wine_DF.shape)
#178 observations and 14 columns


# ## <font color='purple'>The dataset has 178 observations and 14 columns </font>

# In[148]:


#As it is a classification problem by using pairplot we can see three different classes classified based on the Cultivar
import seaborn as sns
sns.pairplot(Wine_DF, hue = 'Cultivar', palette="husl")


# In[8]:


#From the graph above we can see that the data is not hugely overlapping between so we can use the Logistic Regression model for our classification problem


# In[9]:


#Looking at the features
Wine_Features = Wine_DF.columns
print("The features (or attributes) recorded  are :", Wine_Features)


# ## <font color='purple'>The 13 chemical characteristics of wines are 'Cultivar', 'Alcohol', 'MalicAcid', 'Ash', 'Alkalinity', 'Magnesium','Phenols', 'Flavanoids', 'NonFlavanoids', 'Pcyanins', 'ColorIntensity','Hue', 'OD280', 'Proline' </font>

# In[10]:


#Displaying the datatypes of each column
Wine_DF.info()


# # 3.	Keeping your preliminary decisions from (2) in mind, peruse the dataset to:
# # a. Display the datatype of each of the 14 columns to determine if any of the columns need to be transformed to comply with the requirements of your chosen algorithm. Specify the names of columns that require transformation along with the transformation that need to be performed. Include a reasonable explanation as to why the columns need to be transformed as well as what appropriate transformation will be necessary to make the feature algorithm-compliant.
# 

# In[11]:


#Displaying the datatypes of the 14 columns
Wine_DF.dtypes


# ## <font color='purple'>The dataset has all the features in numeric type and only the label variable Cultivar in object type </font>

# ## <font color='purple'>As the label column has object datatype converting it into its actual data type </font>

# In[12]:


#Converting Cultivar dataframe to display actual datatypes
Wine_DF = Wine_DF.convert_dtypes()


# In[13]:


#After conversion looking at the datatypes
Wine_DF.dtypes


# ## <font color='purple'>Logistic Regression requires that variables (feature and target) all need to be numeric (int, double, or float). Target field(Cultivar) in our dataset is of string datatype. Therefore, we will need to convert the variables into their numeric representations.
# ### <font color='blue'> Label variable, Cultivar, is categorical with three categories. So, all we need to do is convert Cultivar1, Cultivar2, Cultivar3 values into appropriate numeric values such as 0/1/2 - This can be handled easily with Sklearn's Label Encoding function.</font>

# In[14]:


#Transforming the target variable to numeric type for further analysis


# In[15]:


from sklearn.preprocessing import LabelEncoder
#Creating an instance of the LabelEncoder class
label_encode  = LabelEncoder()            
#Apply the label_encode to fit_transform the Cultivar column, by creating a new column named 'Cultivar_Type'
Wine_DF['Cultivar_Type'] = label_encode.fit_transform(Wine_DF['Cultivar'])


# In[16]:


#Label Encoder acts as a mapping function or String Indexer and generates corresponding numbers to the string value


# In[17]:


#Looking at data after encoding
Wine_DF.dtypes


# ## b.Identify any other data cleanup and pre-processing that may be required to get the data ready for your chosen machine learning algorithm. This may include handling missing values. Missing values for any feature are to be replaced with a median value for that feature. State so if missing values are not indicated.

# In[18]:


#Checking for null values
Wine_DF.isnull().sum()


# ## <font color='purple'> No null values or missing values are present in the data, so no replacement of values is needed </font>

# In[19]:


#target Class distribution
Wine_DF.groupby(['Cultivar']).size()


# In[20]:


#Label Encoded column distribution
Wine_DF.groupby(['Cultivar_Type']).size()


# In[21]:


################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (48) -- (71/115)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%
#Fairly balaced


# ## <font color='purple'> From above size of function we can see there is a mis balance in our data as there are more records of Cultivar type2(Cultivar_Code-1(71) </font>

# ## <font color='purple'>We have a an unbalanced dataset which means that when the model is trained, it is going to learn more from Cultivar2. This will create bias that may come in when the dataset is not balanced. The bias comes since the model is trained with more samples of one case than the other. This may lead the model to predict more one case over another, thus mispredicting some values of the other case.So we have two options to balance the data. One is to eliminate the excess number of samples from the class that has a higher number of samples (if and only if this does not reduce theoverall dataset size significantly) or add samples of category with lower number. </font>

# ## <font color='purple'>So, we chose to eliminate the excess number of samples from the class that has a higher number of samples Cultivar1 type 0 and Cultivar1 1 </font>

# ## <font color='purple'>We have a fairly balanced dataset as it does not have huge imbalance in the distribution of Cultivar Types and also due less amount of data.We prefer using all data for training the model rather than elminating few data as the model needs to learn enough from the data to make correct predictions </font>

# In[22]:


#So, if we chose to eliminate the excess number of samples from the class that has a higher number of samples Cultivar1 type 0 and Cultivar2 1
#Type_0 = len(Wine_DF[Wine_DF['Cultivar_Code']==0])
#Type_1 = len(Wine_DF[Wine_DF['Cultivar_Code']==1])
#Type_2 = len(Wine_DF[Wine_DF['Cultivar_Code']==2])
#Balanced_Wine_DF = pd.concat( [Wine_DF[Wine_DF['Cultivar_Code']==0].sample(Type_2) ,Wine_DF[Wine_DF['Cultivar_Code']==1].sample(Type_2), Wine_DF[Wine_DF['Cultivar_Code']==2]])
#print(len(Balanced_Wine_DF))


# In[23]:


#Balanced_Wine_DF 


# In[24]:


#Balanced_Wine_DF.groupby('Cultivar_Code').size()


# In[25]:


#Looking at the balanced data
#Balanced_Wine_DF.head() 
#Using above code all the cultivars would be balance to 48 whic is type 3 count


# In[26]:


Model_DF = Wine_DF[['Alcohol','MalicAcid','Ash','Alkalinity','Magnesium','Phenols','Flavanoids','NonFlavanoids','Pcyanins','ColorIntensity','Hue','OD280','Proline','Cultivar_Type']]


# In[27]:


Model_DF.head() 


# In[28]:


#Interpreting the distribution(balance) of data visually
import pandas.util.testing as tm
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = 'Cultivar_Type', data = Model_DF , palette = "husl")
plt.show()
#Around 60 is the count of type 0
##Around 70 is the count of type 1
#Around 50 is the count of type 2


# # 4.	Perform preliminary exploratory data analysis (EDA) pertinent to the problem statement and your chosen machine learning algorithm in (2). This may include basic statistics, data shape, grouping on the outcome variable, generating scatter plots or line plots, etc. as appropriate based on your chosen algorithm. Anything that can give you further insight into your dataset vis-à-vis the machine learning algorithm you have selected should be included with an explanation/conclusion of the output.

# In[29]:


#Exploratory Data Analysis


# In[30]:


Model_DF.describe()


# In[31]:


#The same count number in all the columns indicate every column has same number of values and no presence of missing values
#From min value we can see no column in the has zero values 
#From max value we can understand the highest value of each chemical constituent(feature or column)


# # <font color='blue'> Let us understand each column with respect to its statistics with our three Cultivar types </font>

# In[32]:


#Considering statistics of Alcohol we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Alcohol'].describe()


# ## <font color='purple'> Alcohol </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 
# 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            13.744746                                  12.278732                                  13.153750
# #Minimum                  12.85                                      11.03                                      12.20
# #Max                      14.83                                      13.86                                      14.34                               

# In[33]:


#Lets see individual correlation of alcohol with our label Cultivar_Type(It wouldnt be highly correlated as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Alcohol']].corr()


# In[34]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Alcohol",data=Model_DF)


# In[35]:


#Almost three types are having nearly same spread of datapoints related to alcohol due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[36]:


#Considering statistics of MalicAcid we can understand the following details 
Model_DF.groupby('Cultivar_Type')['MalicAcid'].describe()


# ## <font color='purple'> MalicAcid </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            2.010678                                  1.932676                                  3.333750
# #Minimum                  1.35                                      0.74                                      1.24
# #Max                      4.04                                      5.80                                      5.65   

# In[37]:


#Lets see individual correlation of MalicAcid with our label Cultivar_Type
Model_DF[['Cultivar_Type','MalicAcid']].corr()


# In[38]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="MalicAcid",data=Model_DF)


# In[39]:


#Almost three types are having nearly same spread of datapoints related to Malicacid due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[40]:


#Considering statistics of Ash we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Ash'].describe()


# ## <font color='purple'> Ash </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            2.455593                                  2.244789                                  2.437083
# #Minimum                  2.04                                      1.36                                      2.10
# #Max                      3.22                                      3.23                                      2.86   

# In[41]:


#Lets see individual correlation of Ash with our label Cultivar_Type(It wouldnt be much as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Ash']].corr()


# In[42]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Ash",data=Model_DF)


# In[43]:


#Almost three types are having nearly same spread of datapoints related to Ash due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[44]:


#Considering statistics of Alkalinity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Alkalinity'].describe()


# ## <font color='purple'> Alkalinity </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ##  
#                      Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            17.037288                                 20.238028                                 21.416667(significant variation)
# #Minimum                  11.2                                      10.6                                      17.5     (significant variation)
# #Max                      25.0                                      30.0                                      27.0     (significant variation)  

# In[45]:


#Lets see individual correlation of Alkalinity with our label Cultivar_Type(It wouldnt be much as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Alkalinity']].corr()


# In[46]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Alkalinity",data=Model_DF)


# In[47]:


#The three types are having different spread of datapoints(highest value,lowest values) related to alkalinity due to which the feature could be significant in differentiating between the CultivarTypes


# In[48]:


#Considering statistics of Alkalinity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Magnesium'].describe()


# ## <font color='purple'> Magnesium </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            106.338983                                 94.549296                                 99.312500
# #Minimum                  89.0                                       70.0                                      80.0     
# #Max                      132.0                                      162.0                                     123.0     

# In[49]:


#Lets see individual correlation of Magnesium with our label Cultivar_Type
Model_DF[['Cultivar_Type','Magnesium']].corr()


# In[50]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Magnesium",data=Model_DF)


# In[51]:


#Almost three types are having nearly same spread of datapoints related to Magnesium due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[52]:


#Considering statistics of Phenols we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Phenols'].describe()


# ## <font color='purple'> Phenols </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            2.840169                                 2.258873                                 1.678750   (significant variation)
# #Minimum                  2.20                                      1.10                                      0.98     (significant variation)
# #Max                      3.88                                      3.52                                      2.80     (significant variation)  

# In[53]:


#Lets see individual correlation of Phenols with our label Cultivar_Type
Model_DF[['Cultivar_Type','Phenols']].corr()


# In[54]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Phenols",data=Model_DF)


# In[55]:


#The three types are having different spread of datapoints(highest value,lowest values) related to phenols due to which the feature could be significant in differentiating between the CultivarTypes


# In[56]:


#Considering statistics of Flavanoids we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Flavanoids'].describe()


# ## <font color='purple'> Flavanoids </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            2.982373                                 2.080845                                 0.781458(significant variation)
# #Minimum                  2.19                                      0.57                                      0.34   (significant variation)  
# #Max                      3.93                                      5.08                                      1.57    (significant variation)  

# In[57]:


#Lets see individual correlation of Flavanoids with our label Cultivar_Type
Model_DF[['Cultivar_Type','Flavanoids']].corr()


# In[58]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Flavanoids",data=Model_DF)


# In[59]:


#The three types are having different spread of datapoints(highest value,lowest values) related to Flavanoids due to which the feature could be significant in differentiating between the CultivarTypes


# In[60]:


#Considering statistics of NonFlavanoids we can understand the following details 
Model_DF.groupby('Cultivar_Type')['NonFlavanoids'].describe()


# ## <font color='purple'> NonFlavanoids </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
# 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            0.290000                                  0.363662                                  0.447500(significant variation)
# #Minimum                  0.17                                      0.13                                      0.17     (significant variation)
# #Max                      0.50                                      0.66                                      0.63     (significant variation)  

# In[61]:


#Lets see individual correlation of NonFlavanoids with our label Cultivar_Type
Model_DF[['Cultivar_Type','NonFlavanoids']].corr()


# In[62]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="NonFlavanoids",data=Model_DF)


# In[63]:


#Almost three types are having nearly same spread of datapoints related to NonFlavanods due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[64]:


#Considering statistics of Pcyanins we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Pcyanins'].describe()


# ## <font color='purple'> Pcyanins </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
# 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            1.899322                                 1.630282                                 1.153542
# #Minimum                  1.25                                      0.41                                      0.55     
# #Max                      2.96                                      3.58                                      2.70       

# In[65]:


#Lets see individual correlation of Pcyanins with our label Cultivar_Type
Model_DF[['Cultivar_Type','Pcyanins']].corr()


# In[66]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Pcyanins",data=Model_DF)


# In[67]:


#Almost three types are having nearly same spread of datapoints related to Pcyanins due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[68]:


#Considering statistics of ColorIntensity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['ColorIntensity'].describe()


# ## <font color='purple'> ColorIntensity </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ##  
#                      Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            5.528305                                 3.086620                                 7.396250
# #Minimum                  3.52                                      1.28                                      3.85     
# #Max                      8.9                                       6.0                                       13.0      

# In[69]:


#Lets see individual correlation of ColorIntensity with our label Cultivar_Type
Model_DF[['Cultivar_Type','ColorIntensity']].corr()


# In[70]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="ColorIntensity",data=Model_DF)


# In[71]:


#Almost three types are having nearly same spread of datapoints related to ColorIntensity due to which the feature is not highly significant in differentiating between the CultivarTypes


# In[72]:


#Considering statistics of Hue we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Hue'].describe()


# ## <font color='purple'> Hue </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            1.062034                                 1.056282                                 0.682708(significant variation)
# #Minimum                  0.82                                      0.69                                      0.48     (significant variation)
# #Max                      1.28                                      1.71                                      0.96     (significant variation)  

# In[73]:


#Lets see individual correlation of Hue with our label Cultivar_Type
Model_DF[['Cultivar_Type','Hue']].corr()


# In[74]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Hue",data=Model_DF)


# In[75]:


#The three types are having different spread of datapoints(highest value,lowest values) related to Hue due to which the feature could be significant in differentiating between the CultivarTypes


# In[76]:


#Considering statistics of OD280 we can understand the following details 
Model_DF.groupby('Cultivar_Type')['OD280'].describe()


# ## <font color='purple'> OD280 </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ##                       
#                       Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            3.157797                                 2.785352                                 1.683542(significant variation)
# #Minimum                  2.51                                      1.59                                      1.27     (significant variation)
# #Max                      4.00                                      3.69                                      2.47     (significant variation)  

# In[77]:


#Lets see individual correlation of OD280 with our label Cultivar_Type
Model_DF[['Cultivar_Type','OD280']].corr()


# In[78]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="OD280",data=Model_DF)


# In[79]:


#The three types are having different spread of datapoints(highest value,lowest values) related to OD280 due to which the feature could be significant in differentiating between the CultivarTypes


# In[80]:


#Considering statistics of Proline we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Proline'].describe()


# ## <font color='purple'> Proline </font>
# ## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

# ## 
#                         Cultivar1                                 Cultivar2                                  Cultivar3  
# #No.of values             59                                         71                                         48
# #Average value            1115.711864                               519.507042                                629.895833(significant variation)
# #Minimum                  680.0                                     278.0                                    415.0     (significant variation)
# #Max                      1680.0                                    985.0                                   880.0     (significant variation)  

# In[81]:


#Lets see individual correlation of Proline with our label Cultivar_Type
Model_DF[['Cultivar_Type','Proline']].corr()


# In[82]:


#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Proline",data=Model_DF)


# In[83]:


#The three types are having different spread of datapoints(highest value,lowest values) related to Proline due to which the feature could be significant in differentiating between the CultivarTypes


# In[ ]:





# In[84]:


#sns.catplot(x="Cultivar_Type", y="Alcohol", kind="box",data=Model_DF.sort_values("Cultivar_Type"))


# In[85]:


#sns.catplot(x="Cultivar_Type", y="Alcohol", hue="Cultivar_Type", kind="box", data=Model_DF);


# In[86]:


#Checking the correlation of each feature with target variable by observing the last column in heat map


# In[87]:


Selected_features = ['Alcohol','MalicAcid','Ash','Alkalinity','Magnesium','Phenols','Flavanoids','NonFlavanoids','Pcyanins','ColorIntensity','Hue','OD280','Proline','Cultivar_Type']
X = Model_DF[Selected_features]

plt.subplots(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()


# ## <font color='purple'>We can see that six features are highly correlated above positive (0.5) - 1 and above negative (0.5) -5 </font>

# In[88]:


#On the basis of individual correlation coefficients, we are determining which independent variables are useful in predicting the target value 
#Correlation coefficient value ranges from -1 to +1; closer to 1, stronger the relationship. 
#Also, only correlation coefficients greater than 0.5 in magnitude are considered for further inclusion in the model.
#These variables are considered relevant attributes for prediction of Cultivar Type.


# In[89]:


#For more clear values lets see correlation matrix instead of heatmap


# In[90]:


Model_DF.corr()


# ## <font color='purple'>Flavonoids is the first high negatively correlated value: -0.847498  </font>
# ## <font color='purple'>OD280 is the second high negatively correlated value: -0.788230  </font>
# ## <font color='purple'>Phenols is the third high negatively correlated value: -0.719163 </font>
# ## <font color='purple'>Hue is the fourth high negatively correlated value: -0.617369  </font>
# ## <font color='purple'>Proline is the fifth high negatively correlated value: -0.633717  </font>
# ## <font color='purple'>Alkalinity  is the sixth positively correlated value: 0.517859  </font>
# 

# In[91]:


#Checking the distribution again
Model_DF.groupby(['Cultivar_Type']).size()


# In[92]:


Model_DF.head()


# ## <font color='purple'>Considering only highly correlated features Flavanoids,OD280,Phenols,Hue,Proline,Alkalinity for training and testing the model </font>

# In[93]:


Wine_Corr_features_DF = Model_DF[['Alkalinity','Phenols','Flavanoids','Hue','OD280','Proline']]


# In[94]:


#Input data to model (features)
Wine_Corr_features_DF.head()


# In[95]:


Wine_Target_DF = Model_DF['Cultivar_Type']


# In[96]:


#Label data to model (target variable)
Wine_Target_DF.head()


# ## Question5.	If your chosen algorithm demands training and test datasets, split your wine dataset using an 80/20 split. If dataset is split, evaluate your training and test datasets to ensure they are representative of your full data set. 

# ## <font color='purple'>To Train and Test the Logistic Regression Model, split dataset 80-20%</font>

# In[97]:


#Importing the train test split function from sklearn
from sklearn.model_selection import train_test_split


# In[98]:


#Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(Wine_Corr_features_DF,Wine_Target_DF, test_size=0.20, random_state = 1)


# In[99]:


#Loking at train data
X_train


# In[100]:


#Checking the shape of all splits
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[101]:


# Get a tuple of unique values & their frequency in numpy array for trainig data
uniqueValues, occurCount = np.unique(Y_train, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[102]:


#Total records split for training is 142
#Number of Cultivar1 type in training data are (45) -- (45/142)-- 31.69%
#Number of Cultivar2 type in training data are (58) -- (58/142)-- 40.84% 
#Number of Cultivar3 type in training data are (39) -- (39/142)-- 27.46%
################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (71) -- (71/115)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%


# In[103]:


#displaying training datasets
print(X_train,Y_train)


# In[104]:


# Get a tuple of unique values & their frequency in numpy array for testing data
uniqueValues, occurCount = np.unique(Y_test, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[105]:


#Total records split for testing is 36
#Number of Cultivar1 type in training data are (14) -- (14/36)-- 38.88%
#Number of Cultivar2 type in training data are (13) -- (13/36)-- 36.11% 
#Number of Cultivar3 type in training data are (9) -- (9/36)-- 25.00%
################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (71) -- (71/178)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/178)-- 26.96%


# ## <font color='orange'>Total records split for training is 142</font>
# 
# ### <font color='purple'>Number of Cultivar1 type in training data are (45) -- (45/142)-- 31.69%</font>
# ### <font color='purple'>Number of Cultivar2 type in training data are (58) -- (58/142)-- 40.84%</font>
# ### <font color='purple'>Number of Cultivar3 type in training data are (39) -- (39/142)-- 27.46%</font>
# 
# ## <font color='orange'>Total records split for testing is 36</font>
# ### <font color='purple'>Number of Cultivar1 type in training data are (14) -- (14/36)-- 38.88%</font>
# ### <font color='purple'>Number of Cultivar2 type in training data are (58) -- (13/36)-- 36.11%</font>
# ### <font color='purple'>Number of Cultivar3 type in training data are (39) -- (9/136)-- 25.00%</font>
#         
# ## <font color='orange'> Overall records are 178 </font>      
# ### <font color='purple'>Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%</font>
# ### <font color='purple'>Number of Cultivar2 type in training data are (71) -- (71/115)-- 39.88%</font>
# ### <font color='purple'>Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%</font>
# 
# # <font color='blue'>The training dataset split is representative of overall data as there is only 1 or 2 percents split variation compared to percentages of overall data</font> 
# # <font color='blue'>The testing dataset split is also slightly representative of overall data as there is 4 or 5 percents split variation compared to percentages of overall data</font>

# In[106]:


#Import Logistic regression from SKLearn Libraries
from sklearn.linear_model import LogisticRegression


# # Question6.Use the relevant portion of your dataset to train the model of your selected machine learning algorithm. Do all the necessary preprocessing to determine the parameters for your selected algorithm. For example, you will need to specify (and justify) the number of clusters if you choose to use KMeans clustering algorithm via the Elbow curve, Silhouette analysis, etc. 

# In[107]:


#Creting an instance of Logistic regression using multi_class function
log_reg = LogisticRegression(solver='newton-cg',multi_class='multinomial')
#Applying training data to the model
log_reg.fit(X_train,Y_train)


# In[108]:


#Generating training predictions


# In[109]:


#Generate predictions to evaluate the trained model using X_Train data. 
Ytrain_predict = log_reg.predict(X_train)


# In[110]:


# Our resulting Y_predict variable is of shape (142,). so, it needs to be converted to (142,1) 2-D array
# to a new result dataframe
predict_ytrain = Ytrain_predict.reshape(-1,1)
print(predict_ytrain.shape)


# In[111]:


# The Y_train contains values of our target variable (Cultivar). There were 142 records in the training data. So, Y_train
# has a shape of (142,) - one dimesional. We'll need to reshape this into 2-D (142 rows, 1 column of Y-values)

train_y = (Y_train.values).reshape(-1,1)
print(train_y.shape)
print(train_y.size)


# In[112]:


# We need to obtain probabilities for our predictions. For this, we need to use predict_proba() function of the 
# logistic regression model we instantiated earlier during model training and predictions. 

train_predicted_probs = log_reg.predict_proba(X_train)
print(train_predicted_probs)


# # As mentioned the output of the multicalss logistic regression is set of probabilities based on which the predictions are generated

# In[113]:


#Finally, we add all five variables into a Pandas Dataframe for display purposes. 

np.set_printoptions(suppress=True)  # this is to prevent small values being displayed in scientific notations

train_prob_results_df = pd.DataFrame(train_predicted_probs)
train_prob_results_df["Predicted"] = predict_ytrain
train_prob_results_df["Actual"] = train_y
train_prob_results_df.head(10)


# # 7.Using appropriate metrics for your chosen algorithm, evaluate the trained model. Explain and justify the worthiness of your trained model. 

# In[114]:


# Let's evaluate the trained model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(train_y,predict_ytrain))


# In[115]:


conf_matrix = metrics.confusion_matrix(train_y,predict_ytrain)
conf_matrix


# In[116]:


print("Accuracy:",metrics.accuracy_score(train_y,predict_ytrain))


# In[117]:


print("Precision:",metrics.precision_score(train_y,predict_ytrain, average = "macro"))
print("Recall:",metrics.recall_score(train_y,predict_ytrain,average = "macro"))


# In[118]:


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
auc = multiclass_roc_auc_score(train_y,predict_ytrain, average="macro")
print("Area under curve : ", auc)


# # The model is trained well as the AOC is almost 1.00(0.97) and all other metrics of Accuracy, Recall, Precision,,and F1-score are all above 95% and close 1.0, the highest value for all; metrics. 
# # This means that the trained model couldidentify Cultivar0 as Cultivar0, Cultivar1 as Cultivar1 and  Cultivar2 as Cultivar2 . The model is trained to be able to distinguish between (and therfore, predict correctly) between Cultivar_Types.

# # 8.	Next, use the relevant portion of your dataset (as dictated by the chosen algorithm) to evaluate the performance of your model. Again, use all relevant metrics for your algorithm to discuss the outcome in terms of model’s accuracy and usefulness in generating predictions. These may include such metrics as SSE, MSSE, Silhouette scores, completeness scores, confusion matrix, AOC curve, etc. as dictated by and available for your chosen machine language algorithm.

# In[119]:


#Generate predictions to evaluate the testing model using X_Test data and predicting Y

Ytest_predict  = log_reg.predict(X_test)


# In[120]:


# Our resulting Y_predict variable is of shape (36,). so, it needs to be converted to (36,1) 2-D array
# to a new result dataframe
predict_ytest = Ytest_predict.reshape(-1,1)
print(predict_ytest.shape)


# In[121]:


# The Y_test contains values of our target variable (Cultivar). There were 36 records in the test data. So, Y_test
# has a shape of (36,) - one dimesional. We'll need to reshape this into 2-D (36 rows, 1 column of Y-values)

test_y = (Y_test.values).reshape(-1,1)
print(test_y.shape)
print(test_y.size)


# In[122]:


# We need to obtain probabilities for our predictions. For this, we need to use predict_proba() function of the 
# logistic regression model we instantiated earlier during model training and predictions. 

test_predicted_probs = log_reg.predict_proba(X_test)
print(test_predicted_probs)


# In[123]:


#Finally, we add all three variables into a Pandas Dataframe for display purposes. 

np.set_printoptions(suppress=True)  
import pandas as pd
pd.set_option('display.precision',11)
pd.DataFrame({'2':[.001]}) 

# this is to prevent small values being displayed in scientific notations
test_prob_results_df = pd.DataFrame(test_predicted_probs)
test_prob_results_df["Predicted"] = predict_ytest
test_prob_results_df["Actual"] = test_y
test_prob_results_df.head(40)


# Note columns(0, 1 and 2) represents the probability of classifying an observation as being classified as 0,1 and 2 as it is a multinomial classification Cultivar1, Cultivar2, Cultivar3
# respectively. 
#So, for the first record, the value predicted by the model for Cultivar Type is "2" and the actual Cultivar_Type value for that observation in the test dataset is also "2"
#Probaility that this observation is correctly classified as being 2 is 0.85333574416. While the proabability that is is 
# misclassified as 0 is 0.00124438217 and as 1 is 0.14541987367 respectively

#Judging from these probabilities for all the 36 records, the model is predicting with high level of accuarcy.
#Since high-level here is Hard to say. It can be interpreted correctly by evaluating various metrics. 


# In[124]:


#In case of more data we could have trained the model more better


# In[125]:


# Let's evaluate the model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(Y_test,Ytest_predict))


# In[126]:


conf_matrix = metrics.confusion_matrix(Y_test, Ytest_predict)
conf_matrix


# # Understanding the confusion metrics output of multiclass-classification
#                      
#                        #Predicted values
#                              0   1   2
#                       0  ([[13,  1,  0],
# #Actual values        1    [ 1, 12,  0],
#                       2    [ 0,  0,  9]])
#                       
# Based on the Confusion Matrix, we can see that the predictive performance of the trained model is very
# good. Of the total 36 cases, 94.4% are predicted correctly (36.11% Cultivar0, 33.33% Cultivar1 and 25% Cultivar2 ) while only
# 5.6% are incorrectly predicted. This is further supported by the fact that Recall,Accuracy, Precision, and
# F1-score is >= 0.94, very close to 1.0, the ideal score. 

# In[127]:


print("Accuracy:",metrics.accuracy_score(Y_test, Ytest_predict))


# In[128]:


print("Precision:",metrics.precision_score(Y_test,Ytest_predict, average = "macro"))
print("Recall:",metrics.recall_score(Y_test,Ytest_predict,average = "macro"))


# In[129]:


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
auc = multiclass_roc_auc_score(Y_test, Ytest_predict, average="macro")
print("Area under curve : ", auc)


# # The AOC is also close to 1.0. Hence, the trained Logistic Regression model is useful in accuaretly predicting the Cultivar type based on the chemical characteristics of wine

# 
#             ## ######################################################################
#             Accuracy of the model is : 94.4# percent(The model can calculate 94% accurately)
#             ######################################################################
#             Recall score of the complete model is 95%
#             Recall score of predicting Cultivar = 0   is  93%
#             Recall score of predicting Cultivar = 1   is  92%
#             Recall score of predicting Cultivar = 2   is 100%
#             #####################################################################
#             Precision score of the complete model is 95%
#             Precision score of predicting Cultivar = 0   is  93%
#             Precision score of predicting Cultivar = 1   is  92%
#             Precision score of predicting Cultivar = 2   is 100%
#             #####################################################################
#             F1 score of the complete model is 95%
#             F1 score of predicting Cultivar = 0   is  93%
#             F1 score of predicting Cultivar = 1   is  92%
#             F1 score of predicting Cultivar = 2   is 100%
#             #####################################################################

#         ## #####
#         # Area Under the Curve is calculated =0.96
#         # Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s and 2's as 2's. By analogy, Higher the AUC, better the model is at distinguishing between Cultivar Types under three to three categories
#         #As our Classification, auc score is 0.96 i.e; the model can classify between 0's, 1's,2's and predict them with around 94% accuracy
#         #This indicates our model is worthy and also a good model in predicting the survival Cultivar categories

#        ## #####################################################################
#        #Recall:
#         It talks about the quality of the machine learning model when it comes
#         to predicting a positive class. So out of total positive classes, how many
#         was the model able to predict correctly? This metric is widely used as
#         evaluation criteria for classification models.
#         The recall values of our model are
#         ######################################################################
#             Recall score of the complete model is 95%
#             Recall score of predicting Cultivar = 0   is  93%
#             Recall score of predicting Cultivar = 1   is  92%
#             Recall score of predicting Cultivar = 2   is 100%
#         #which is good
#         #####################################################################
#         #Precision:
#         Precision is about the number of actual positive cases out of all the positive
#         cases predicted by the model
#         The precision values of our model are
#         #####################################################################
#             Precision score of the complete model is 95%
#             Precision score of predicting Cultivar = 0   is  93%
#             Precision score of predicting Cultivar = 1   is  92%
#             Precision score of predicting Cultivar = 2   is 100%
#         #which are good
#         ################################################################
#         #F1 Score:
#         It considers both the precision p and the recall r of the test to compute the score
#         The F1score values of our model are
#         #which are good
#         #####################################################################
#             F1 score of the complete model is 95%
#             F1 score of predicting Cultivar = 0   is  93%
#             F1 score of predicting Cultivar = 1   is  92%
#             F1 score of predicting Cultivar = 2   is 100%
#         #####################################################################
#         # Area Under the Curve is calculated =0.9604525908873734
#         # Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between Cultivartypes that
#         #As our Classification, auc score is 0.96 i.e; the model can classify between 0's, 1's, 2's and predict them with around 94# accuracy
#         #This indicates our model is worthy and also a good model in predicting the Cultivar types
#         ########################################################################
#         Accuracy of the model is :94.4 percent(The model can calculate 94% accurately)

# # In case of non linear data and high overlapping between the target classes we can use the following algorithms(Decision Tree and Random Forest) for classification when complexity and time are not a matter of issue(Optional)

# In[130]:


# Checking Decision Tree Classifier


# In[131]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics


# In[132]:


#Creating an instance of Decision tree classifier
DTC = DecisionTreeClassifier()

# Apllying Decision Tree Classifer on the training data
DTC = DTC.fit(X_train,Y_train)

DTC_prob = DTC.predict_proba(X_test)
#Predict the response for test dataset
DTC_y_pred = DTC.predict(X_test)


# In[133]:


#Looking at the predictions
DTC_y_pred


# In[134]:


#Creating a pandas dataframe for Decision tree result
DTC_results_df = pd.DataFrame(DTC_prob)
test_prob_results_df["Predicted"] = DTC_y_pred
test_prob_results_df["Actual"] = test_y
test_prob_results_df.head(20)


# In[135]:


print("Accuracy:",metrics.accuracy_score(Y_test, DTC_y_pred))
#Clearly has less accuracy compared to logistic regression model


# In[136]:


# Let's evaluate the model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(Y_test,DTC_y_pred))


# In[137]:


#The precision, f1-score, recall are also less compared to logistic regression model


# In[138]:


auc = multiclass_roc_auc_score(Y_test,DTC_y_pred, average="macro")
print("Area under curve : ", auc)
#Auc is less when compared to Logistic regression but 0.9 is not a bad value


# In[139]:


# Random Forest is an ensemble technique which uses decision trees as the base learners it has if else loops which are time consuming


# In[140]:


# Checking Random Forest Classifier


# In[141]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
RFC = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
RFC.fit(X_train,Y_train)


# In[142]:


# Actual class predictions
RF_predictions = RFC.predict(X_test)
# Probabilities for each class
RF_probs = RFC.predict_proba(X_test)[:, 1]


# In[143]:


#Creating a pandas dataframe for Decision tree result
DTC_results_df = pd.DataFrame(RF_probs)
test_prob_results_df["Predicted"] = RF_predictions 
test_prob_results_df["Actual"] = test_y
test_prob_results_df.head(20)


# In[144]:


print("Accuracy:",metrics.accuracy_score(Y_test,RF_predictions))
#Random forest also has accuracy value similar to logistic regression model


# In[145]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test,RF_predictions))
print(classification_report(Y_test,RF_predictions))
print(accuracy_score(Y_test,RF_predictions))


# In[146]:


auc = multiclass_roc_auc_score(Y_test,RF_predictions, average="macro")
print("Area under curve : ", auc)


# In[147]:


# All the values like accuracy, f1score, recall, precision are same as logistic regression outputs which makes random forest also a good model to classify the Cultivar_Types

