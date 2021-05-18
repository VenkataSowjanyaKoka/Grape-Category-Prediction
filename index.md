```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>


# <font color='black'>Question1.What exactly is the business problem you are trying to solve? Summarize this in the form of a meaningful problem statement? </font>

## <font color='purple'>Determining the category or variety of grape used in making wines based on several chemical characteristics of individual wines</font>

# <font color='orange'> Problem Statement</font>

## <font color='purple'>Predicting the category or variety of grape used in making wines based on the chemical composition of 13 constituents found in each of the three types of grape</font>



```python
# Data Required (We need to have data related to charateristics based on which a particular class is assigned to train our model)
```

# Question2. What are some of preliminary decisions you may need to make based on your problem statement? Your answer should include identification of an initial machine learning algorithm you will apply with respect to the problem statement in (1). Justification should be based on identification of the category of machine learning (supervised, unsupervised, etc.) as well as suggested machine learning algorithm from within the identified machine learning category. 

## <font color='purple'>The business problem we are trying to solve is predicting the type of wine where we are dealing with labeled data.(Cultivar) </font>


# <font color='orange'> Identification of the category of Machine Learning</font>

## <font color='purple'>As we have a classified data where there is a target class(Cultivar) and some characteristics based on which this target is classified, we need to train the model to perform same process </font>

## <font color='purple'>The category of machine learning which deals with labeled data is Supervised Learning </font>


## <font color='purple'>In Supervised learning both input features  and target variable are available for each training data. A supervised learning algorithm then tries to learn the relationship between the input and output variables from the data, so that when input x is given, it is able to produce the corresponding output y. And to do so, the algorithm iterates the training data to adjust the parameters of a model, until it could generalize the data. This is called the learning process. </font>


## <font color='purple'>As we are trying to determine(predict) the category of grape which is a labeled variable we need to choose between the sub division of Supervised Learning  </font>
## <font color='orange'>Regression and Classification</font>
## <font color='purple'>Regression is used when the target variable is numeric and continuous </font>
## <font color='purple'>Classification is used when the target variable is categorical </font>
## <font color='purple'>Our problem has target variable as categorical type (Type of grape) </font>

## <font color='purple'>So under the types of supervised learning algorithms we would eliminate the choice of Linear Regression as it deals with continuous variables  </font>

## <font color='purple'>We need to choose from the further available alogorithms Logistic Regression, Decision Tree and Random Forest</font>

# <font color='orange'> Identification of the category of Machine Learning Algorithm </font>

## <font color='purple'>While all three of the remaining algorithms are applicable to achieve the objective, we will start with Logistic Regression.</font>

## <font color='purple'>The time consumed and complexity of logistic regression model is less when compared to Decision Trees and Random Forests. As there will be creation a multiple branches or if else loops created in case of Decision trees and random forests resulting more time consumption for training the model </font>

## <font color='purple'>As there are three different categories in the target variable so we choose Multinomial Logistic Regression which classifies based on one versus rest method</font>

### In Multi-class logistic regression creates different groups using one versus ret methodology, 
### For example Cultivar1 class the outputs are considered as 1,-1,-1 for Cultivar1, Cultivar2, Cultivar3 respectively  
### In case of Cultivar2 class the outputs are considered as -1,1,-1 for Cultivar1, Cultivar2, Cultivar3 respectively
### Similarly for Cultivar3 is as -1,-1,1 for Cultivar1, Cultivar2, Cultivar3 respectively
### So after this the first model is created as M1 based on input features and the first column outputs for example from above conditions 1,-1,-1 and 
### this model(M1) will be able to predict if the output is Cultivar1 or not
### Similarly M2 model  will be created for Cultivar2 as output
### Similarly M3 model  will be created for Cultivar3 as output

### when test data is given then the output would be calculated from three model M1, M2, M3 (probabilities as otputs)
### Were sum of the three  probabilites is equal to 1
### For generation og prediction the array of three probabilities is considered and the one with highest probability is considered to be the prediction output
### So if the output probabilities are [0.25, 0.25, 0.5] the output would be Cultivar3


```python
#Loading the pandas library
import pandas as pd
```


```python
#Loading the numpy library
import numpy as np
```


```python
#Loading sklearn for machine learning packages
import sklearn
```


```python
## importing the wine dataset with pandas
Wine_DF = pd.read_csv('D://wine.csv', header=0, sep=',')
```


```python
#Seeing the shape of the dataset
print("Shape of the data contained in wine.csv is", Wine_DF.shape)
#178 observations and 14 columns
```

    Shape of the data contained in wine.csv is (178, 14)


## <font color='purple'>The dataset has 178 observations and 14 columns </font>


```python
#As it is a classification problem by using pairplot we can see three different classes classified based on the Cultivar
import seaborn as sns
sns.pairplot(Wine_DF, hue = 'Cultivar', palette="husl")

```

    /opt/anaconda3/lib/python3.7/site-packages/statsmodels/nonparametric/kde.py:487: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    /opt/anaconda3/lib/python3.7/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2





    <seaborn.axisgrid.PairGrid at 0x1a26682910>




![png](output_27_2.png)



```python
#From the graph above we can see that the data is not hugely overlapping between so we can use the Logistic Regression model for our classification problem
```


```python
#Looking at the features
Wine_Features = Wine_DF.columns
print("The features (or attributes) recorded  are :", Wine_Features)
```

    The features (or attributes) recorded  are : Index(['Cultivar', 'Alcohol', 'MalicAcid', 'Ash', 'Alkalinity', 'Magnesium',
           'Phenols', 'Flavanoids', 'NonFlavanoids', 'Pcyanins', 'ColorIntensity',
           'Hue', 'OD280', 'Proline'],
          dtype='object')


## <font color='purple'>The 13 chemical characteristics of wines are 'Cultivar', 'Alcohol', 'MalicAcid', 'Ash', 'Alkalinity', 'Magnesium','Phenols', 'Flavanoids', 'NonFlavanoids', 'Pcyanins', 'ColorIntensity','Hue', 'OD280', 'Proline' </font>


```python
#Displaying the datatypes of each column
Wine_DF.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Cultivar        178 non-null    object 
     1   Alcohol         178 non-null    float64
     2   MalicAcid       178 non-null    float64
     3   Ash             178 non-null    float64
     4   Alkalinity      178 non-null    float64
     5   Magnesium       178 non-null    int64  
     6   Phenols         178 non-null    float64
     7   Flavanoids      178 non-null    float64
     8   NonFlavanoids   178 non-null    float64
     9   Pcyanins        178 non-null    float64
     10  ColorIntensity  178 non-null    float64
     11  Hue             178 non-null    float64
     12  OD280           178 non-null    float64
     13  Proline         178 non-null    int64  
    dtypes: float64(11), int64(2), object(1)
    memory usage: 19.6+ KB


# 3.	Keeping your preliminary decisions from (2) in mind, peruse the dataset to:
# a. Display the datatype of each of the 14 columns to determine if any of the columns need to be transformed to comply with the requirements of your chosen algorithm. Specify the names of columns that require transformation along with the transformation that need to be performed. Include a reasonable explanation as to why the columns need to be transformed as well as what appropriate transformation will be necessary to make the feature algorithm-compliant.



```python
#Displaying the datatypes of the 14 columns
Wine_DF.dtypes
```




    Cultivar           object
    Alcohol           float64
    MalicAcid         float64
    Ash               float64
    Alkalinity        float64
    Magnesium           int64
    Phenols           float64
    Flavanoids        float64
    NonFlavanoids     float64
    Pcyanins          float64
    ColorIntensity    float64
    Hue               float64
    OD280             float64
    Proline             int64
    dtype: object



## <font color='purple'>The dataset has all the features in numeric type and only the label variable Cultivar in object type </font>

## <font color='purple'>As the label column has object datatype converting it into its actual data type </font>


```python
#Converting Cultivar dataframe to display actual datatypes
Wine_DF = Wine_DF.convert_dtypes()
```


```python
#After conversion looking at the datatypes
Wine_DF.dtypes
```




    Cultivar           string
    Alcohol           float64
    MalicAcid         float64
    Ash               float64
    Alkalinity        float64
    Magnesium           Int64
    Phenols           float64
    Flavanoids        float64
    NonFlavanoids     float64
    Pcyanins          float64
    ColorIntensity    float64
    Hue               float64
    OD280             float64
    Proline             Int64
    dtype: object



## <font color='purple'>Logistic Regression requires that variables (feature and target) all need to be numeric (int, double, or float). Target field(Cultivar) in our dataset is of string datatype. Therefore, we will need to convert the variables into their numeric representations.
### <font color='blue'> Label variable, Cultivar, is categorical with three categories. So, all we need to do is convert Cultivar1, Cultivar2, Cultivar3 values into appropriate numeric values such as 0/1/2 - This can be handled easily with Sklearn's Label Encoding function.</font>


```python
#Transforming the target variable to numeric type for further analysis
```


```python
from sklearn.preprocessing import LabelEncoder
#Creating an instance of the LabelEncoder class
label_encode  = LabelEncoder()            
#Apply the label_encode to fit_transform the Cultivar column, by creating a new column named 'Cultivar_Type'
Wine_DF['Cultivar_Type'] = label_encode.fit_transform(Wine_DF['Cultivar'])
```


```python
#Label Encoder acts as a mapping function or String Indexer and generates corresponding numbers to the string value
```


```python
#Looking at data after encoding
Wine_DF.dtypes
```




    Cultivar           string
    Alcohol           float64
    MalicAcid         float64
    Ash               float64
    Alkalinity        float64
    Magnesium           Int64
    Phenols           float64
    Flavanoids        float64
    NonFlavanoids     float64
    Pcyanins          float64
    ColorIntensity    float64
    Hue               float64
    OD280             float64
    Proline             Int64
    Cultivar_Type       int64
    dtype: object



## b.Identify any other data cleanup and pre-processing that may be required to get the data ready for your chosen machine learning algorithm. This may include handling missing values. Missing values for any feature are to be replaced with a median value for that feature. State so if missing values are not indicated.


```python
#Checking for null values
Wine_DF.isnull().sum()
```




    Cultivar          0
    Alcohol           0
    MalicAcid         0
    Ash               0
    Alkalinity        0
    Magnesium         0
    Phenols           0
    Flavanoids        0
    NonFlavanoids     0
    Pcyanins          0
    ColorIntensity    0
    Hue               0
    OD280             0
    Proline           0
    Cultivar_Type     0
    dtype: int64



## <font color='purple'> No null values or missing values are present in the data, so no replacement of values is needed </font>


```python
#target Class distribution
Wine_DF.groupby(['Cultivar']).size()
```




    Cultivar
    cultivar1    59
    cultivar2    71
    cultivar3    48
    dtype: int64




```python
#Label Encoded column distribution
Wine_DF.groupby(['Cultivar_Type']).size()
```




    Cultivar_Type
    0    59
    1    71
    2    48
    dtype: int64




```python
################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (48) -- (71/115)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%
#Fairly balaced
```

## <font color='purple'> From above size of function we can see there is a mis balance in our data as there are more records of Cultivar type2(Cultivar_Code-1(71) </font>

## <font color='purple'>We have a an unbalanced dataset which means that when the model is trained, it is going to learn more from Cultivar2. This will create bias that may come in when the dataset is not balanced. The bias comes since the model is trained with more samples of one case than the other. This may lead the model to predict more one case over another, thus mispredicting some values of the other case.So we have two options to balance the data. One is to eliminate the excess number of samples from the class that has a higher number of samples (if and only if this does not reduce theoverall dataset size significantly) or add samples of category with lower number. </font>

## <font color='purple'>So, we chose to eliminate the excess number of samples from the class that has a higher number of samples Cultivar1 type 0 and Cultivar1 1 </font>

## <font color='purple'>We have a fairly balanced dataset as it does not have huge imbalance in the distribution of Cultivar Types and also due less amount of data.We prefer using all data for training the model rather than elminating few data as the model needs to learn enough from the data to make correct predictions </font>


```python
#So, if we chose to eliminate the excess number of samples from the class that has a higher number of samples Cultivar1 type 0 and Cultivar2 1
#Type_0 = len(Wine_DF[Wine_DF['Cultivar_Code']==0])
#Type_1 = len(Wine_DF[Wine_DF['Cultivar_Code']==1])
#Type_2 = len(Wine_DF[Wine_DF['Cultivar_Code']==2])
#Balanced_Wine_DF = pd.concat( [Wine_DF[Wine_DF['Cultivar_Code']==0].sample(Type_2) ,Wine_DF[Wine_DF['Cultivar_Code']==1].sample(Type_2), Wine_DF[Wine_DF['Cultivar_Code']==2]])
#print(len(Balanced_Wine_DF))
```


```python
#Balanced_Wine_DF 
```


```python
#Balanced_Wine_DF.groupby('Cultivar_Code').size()
```


```python
#Looking at the balanced data
#Balanced_Wine_DF.head() 
#Using above code all the cultivars would be balance to 48 whic is type 3 count
```


```python
Model_DF = Wine_DF[['Alcohol','MalicAcid','Ash','Alkalinity','Magnesium','Phenols','Flavanoids','NonFlavanoids','Pcyanins','ColorIntensity','Hue','OD280','Proline','Cultivar_Type']]
```


```python
Model_DF.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>MalicAcid</th>
      <th>Ash</th>
      <th>Alkalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>NonFlavanoids</th>
      <th>Pcyanins</th>
      <th>ColorIntensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Cultivar_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Interpreting the distribution(balance) of data visually
import pandas.util.testing as tm
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = 'Cultivar_Type', data = Model_DF , palette = "husl")
plt.show()
#Around 60 is the count of type 0
##Around 70 is the count of type 1
#Around 50 is the count of type 2
```


![png](output_59_0.png)


# 4.	Perform preliminary exploratory data analysis (EDA) pertinent to the problem statement and your chosen machine learning algorithm in (2). This may include basic statistics, data shape, grouping on the outcome variable, generating scatter plots or line plots, etc. as appropriate based on your chosen algorithm. Anything that can give you further insight into your dataset vis-à-vis the machine learning algorithm you have selected should be included with an explanation/conclusion of the output.


```python
#Exploratory Data Analysis
```


```python
Model_DF.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>MalicAcid</th>
      <th>Ash</th>
      <th>Alkalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>NonFlavanoids</th>
      <th>Pcyanins</th>
      <th>ColorIntensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Cultivar_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
      <td>0.938202</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
      <td>0.775035</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The same count number in all the columns indicate every column has same number of values and no presence of missing values
#From min value we can see no column in the has zero values 
#From max value we can understand the highest value of each chemical constituent(feature or column)
```

# <font color='blue'> Let us understand each column with respect to its statistics with our three Cultivar types </font>


```python
#Considering statistics of Alcohol we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Alcohol'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>13.744746</td>
      <td>0.462125</td>
      <td>12.85</td>
      <td>13.400</td>
      <td>13.750</td>
      <td>14.100</td>
      <td>14.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>12.278732</td>
      <td>0.537964</td>
      <td>11.03</td>
      <td>11.915</td>
      <td>12.290</td>
      <td>12.515</td>
      <td>13.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>13.153750</td>
      <td>0.530241</td>
      <td>12.20</td>
      <td>12.805</td>
      <td>13.165</td>
      <td>13.505</td>
      <td>14.34</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Alcohol </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 


## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            13.744746                                  12.278732                                  13.153750
#Minimum                  12.85                                      11.03                                      12.20
#Max                      14.83                                      13.86                                      14.34                               


```python
#Lets see individual correlation of alcohol with our label Cultivar_Type(It wouldnt be highly correlated as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Alcohol']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.328222</td>
    </tr>
    <tr>
      <th>Alcohol</th>
      <td>-0.328222</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Alcohol",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a2450ead0>




![png](output_69_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to alcohol due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of MalicAcid we can understand the following details 
Model_DF.groupby('Cultivar_Type')['MalicAcid'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.010678</td>
      <td>0.688549</td>
      <td>1.35</td>
      <td>1.6650</td>
      <td>1.770</td>
      <td>1.9350</td>
      <td>4.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>1.932676</td>
      <td>1.015569</td>
      <td>0.74</td>
      <td>1.2700</td>
      <td>1.610</td>
      <td>2.1450</td>
      <td>5.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>3.333750</td>
      <td>1.087906</td>
      <td>1.24</td>
      <td>2.5875</td>
      <td>3.265</td>
      <td>3.9575</td>
      <td>5.65</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> MalicAcid </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            2.010678                                  1.932676                                  3.333750
#Minimum                  1.35                                      0.74                                      1.24
#Max                      4.04                                      5.80                                      5.65   


```python
#Lets see individual correlation of MalicAcid with our label Cultivar_Type
Model_DF[['Cultivar_Type','MalicAcid']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>MalicAcid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>0.437776</td>
    </tr>
    <tr>
      <th>MalicAcid</th>
      <td>0.437776</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="MalicAcid",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a2461c210>




![png](output_75_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to Malicacid due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Ash we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Ash'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.455593</td>
      <td>0.227166</td>
      <td>2.04</td>
      <td>2.295</td>
      <td>2.44</td>
      <td>2.6150</td>
      <td>3.22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>2.244789</td>
      <td>0.315467</td>
      <td>1.36</td>
      <td>2.000</td>
      <td>2.24</td>
      <td>2.4200</td>
      <td>3.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>2.437083</td>
      <td>0.184690</td>
      <td>2.10</td>
      <td>2.300</td>
      <td>2.38</td>
      <td>2.6025</td>
      <td>2.86</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Ash </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            2.455593                                  2.244789                                  2.437083
#Minimum                  2.04                                      1.36                                      2.10
#Max                      3.22                                      3.23                                      2.86   


```python
#Lets see individual correlation of Ash with our label Cultivar_Type(It wouldnt be much as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Ash']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Ash</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.049643</td>
    </tr>
    <tr>
      <th>Ash</th>
      <td>-0.049643</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Ash",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a247abcd0>




![png](output_81_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to Ash due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Alkalinity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Alkalinity'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>17.037288</td>
      <td>2.546322</td>
      <td>11.2</td>
      <td>16.0</td>
      <td>16.8</td>
      <td>18.7</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>20.238028</td>
      <td>3.349770</td>
      <td>10.6</td>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>21.416667</td>
      <td>2.258161</td>
      <td>17.5</td>
      <td>20.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Alkalinity </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

##  
                     Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            17.037288                                 20.238028                                 21.416667(significant variation)
#Minimum                  11.2                                      10.6                                      17.5     (significant variation)
#Max                      25.0                                      30.0                                      27.0     (significant variation)  


```python
#Lets see individual correlation of Alkalinity with our label Cultivar_Type(It wouldnt be much as there is no significant variation in the values for each type as seen above)
Model_DF[['Cultivar_Type','Alkalinity']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Alkalinity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>0.517859</td>
    </tr>
    <tr>
      <th>Alkalinity</th>
      <td>0.517859</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Alkalinity",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a248a62d0>




![png](output_87_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to alkalinity due to which the feature could be significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Alkalinity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Magnesium'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>106.338983</td>
      <td>10.498949</td>
      <td>89.0</td>
      <td>98.00</td>
      <td>104.0</td>
      <td>114.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>94.549296</td>
      <td>16.753497</td>
      <td>70.0</td>
      <td>85.50</td>
      <td>88.0</td>
      <td>99.5</td>
      <td>162.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>99.312500</td>
      <td>10.890473</td>
      <td>80.0</td>
      <td>89.75</td>
      <td>97.0</td>
      <td>106.0</td>
      <td>123.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Magnesium </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            106.338983                                 94.549296                                 99.312500
#Minimum                  89.0                                       70.0                                      80.0     
#Max                      132.0                                      162.0                                     123.0     


```python
#Lets see individual correlation of Magnesium with our label Cultivar_Type
Model_DF[['Cultivar_Type','Magnesium']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Magnesium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.209179</td>
    </tr>
    <tr>
      <th>Magnesium</th>
      <td>-0.209179</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Magnesium",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a24aaf750>




![png](output_93_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to Magnesium due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Phenols we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Phenols'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.840169</td>
      <td>0.338961</td>
      <td>2.20</td>
      <td>2.6000</td>
      <td>2.800</td>
      <td>3.0000</td>
      <td>3.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>2.258873</td>
      <td>0.545361</td>
      <td>1.10</td>
      <td>1.8950</td>
      <td>2.200</td>
      <td>2.5600</td>
      <td>3.52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>1.678750</td>
      <td>0.356971</td>
      <td>0.98</td>
      <td>1.4075</td>
      <td>1.635</td>
      <td>1.8075</td>
      <td>2.80</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Phenols </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            2.840169                                 2.258873                                 1.678750   (significant variation)
#Minimum                  2.20                                      1.10                                      0.98     (significant variation)
#Max                      3.88                                      3.52                                      2.80     (significant variation)  


```python
#Lets see individual correlation of Phenols with our label Cultivar_Type
Model_DF[['Cultivar_Type','Phenols']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Phenols</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.719163</td>
    </tr>
    <tr>
      <th>Phenols</th>
      <td>-0.719163</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Phenols",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a24c3a590>




![png](output_99_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to phenols due to which the feature could be significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Flavanoids we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Flavanoids'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>2.982373</td>
      <td>0.397494</td>
      <td>2.19</td>
      <td>2.680</td>
      <td>2.980</td>
      <td>3.245</td>
      <td>3.93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>2.080845</td>
      <td>0.705701</td>
      <td>0.57</td>
      <td>1.605</td>
      <td>2.030</td>
      <td>2.475</td>
      <td>5.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>0.781458</td>
      <td>0.293504</td>
      <td>0.34</td>
      <td>0.580</td>
      <td>0.685</td>
      <td>0.920</td>
      <td>1.57</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Flavanoids </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            2.982373                                 2.080845                                 0.781458(significant variation)
#Minimum                  2.19                                      0.57                                      0.34   (significant variation)  
#Max                      3.93                                      5.08                                      1.57    (significant variation)  


```python
#Lets see individual correlation of Flavanoids with our label Cultivar_Type
Model_DF[['Cultivar_Type','Flavanoids']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Flavanoids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.847498</td>
    </tr>
    <tr>
      <th>Flavanoids</th>
      <td>-0.847498</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Flavanoids",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a24c3a2d0>




![png](output_105_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to Flavanoids due to which the feature could be significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of NonFlavanoids we can understand the following details 
Model_DF.groupby('Cultivar_Type')['NonFlavanoids'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>0.290000</td>
      <td>0.070049</td>
      <td>0.17</td>
      <td>0.2550</td>
      <td>0.29</td>
      <td>0.32</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>0.363662</td>
      <td>0.123961</td>
      <td>0.13</td>
      <td>0.2700</td>
      <td>0.37</td>
      <td>0.43</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>0.447500</td>
      <td>0.124140</td>
      <td>0.17</td>
      <td>0.3975</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>0.63</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> NonFlavanoids </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 

                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            0.290000                                  0.363662                                  0.447500(significant variation)
#Minimum                  0.17                                      0.13                                      0.17     (significant variation)
#Max                      0.50                                      0.66                                      0.63     (significant variation)  


```python
#Lets see individual correlation of NonFlavanoids with our label Cultivar_Type
Model_DF[['Cultivar_Type','NonFlavanoids']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>NonFlavanoids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>0.489109</td>
    </tr>
    <tr>
      <th>NonFlavanoids</th>
      <td>0.489109</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="NonFlavanoids",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a24ed0d90>




![png](output_111_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to NonFlavanods due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Pcyanins we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Pcyanins'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>1.899322</td>
      <td>0.412109</td>
      <td>1.25</td>
      <td>1.640</td>
      <td>1.870</td>
      <td>2.090</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>1.630282</td>
      <td>0.602068</td>
      <td>0.41</td>
      <td>1.350</td>
      <td>1.610</td>
      <td>1.885</td>
      <td>3.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>1.153542</td>
      <td>0.408836</td>
      <td>0.55</td>
      <td>0.855</td>
      <td>1.105</td>
      <td>1.350</td>
      <td>2.70</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Pcyanins </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 

                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            1.899322                                 1.630282                                 1.153542
#Minimum                  1.25                                      0.41                                      0.55     
#Max                      2.96                                      3.58                                      2.70       


```python
#Lets see individual correlation of Pcyanins with our label Cultivar_Type
Model_DF[['Cultivar_Type','Pcyanins']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Pcyanins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.00000</td>
      <td>-0.49913</td>
    </tr>
    <tr>
      <th>Pcyanins</th>
      <td>-0.49913</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Pcyanins",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a24ed0210>




![png](output_117_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to Pcyanins due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of ColorIntensity we can understand the following details 
Model_DF.groupby('Cultivar_Type')['ColorIntensity'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>5.528305</td>
      <td>1.238573</td>
      <td>3.52</td>
      <td>4.5500</td>
      <td>5.40</td>
      <td>6.225</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>3.086620</td>
      <td>0.924929</td>
      <td>1.28</td>
      <td>2.5350</td>
      <td>2.90</td>
      <td>3.400</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>7.396250</td>
      <td>2.310942</td>
      <td>3.85</td>
      <td>5.4375</td>
      <td>7.55</td>
      <td>9.225</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> ColorIntensity </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

##  
                     Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            5.528305                                 3.086620                                 7.396250
#Minimum                  3.52                                      1.28                                      3.85     
#Max                      8.9                                       6.0                                       13.0      


```python
#Lets see individual correlation of ColorIntensity with our label Cultivar_Type
Model_DF[['Cultivar_Type','ColorIntensity']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>ColorIntensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>0.265668</td>
    </tr>
    <tr>
      <th>ColorIntensity</th>
      <td>0.265668</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="ColorIntensity",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a25052ed0>




![png](output_123_1.png)



```python
#Almost three types are having nearly same spread of datapoints related to ColorIntensity due to which the feature is not highly significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Hue we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Hue'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>1.062034</td>
      <td>0.116483</td>
      <td>0.82</td>
      <td>0.9950</td>
      <td>1.070</td>
      <td>1.1300</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>1.056282</td>
      <td>0.202937</td>
      <td>0.69</td>
      <td>0.9250</td>
      <td>1.040</td>
      <td>1.2050</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>0.682708</td>
      <td>0.114441</td>
      <td>0.48</td>
      <td>0.5875</td>
      <td>0.665</td>
      <td>0.7525</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Hue </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            1.062034                                 1.056282                                 0.682708(significant variation)
#Minimum                  0.82                                      0.69                                      0.48     (significant variation)
#Max                      1.28                                      1.71                                      0.96     (significant variation)  


```python
#Lets see individual correlation of Hue with our label Cultivar_Type
Model_DF[['Cultivar_Type','Hue']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Hue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.617369</td>
    </tr>
    <tr>
      <th>Hue</th>
      <td>-0.617369</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Hue",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a2534c690>




![png](output_129_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to Hue due to which the feature could be significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of OD280 we can understand the following details 
Model_DF.groupby('Cultivar_Type')['OD280'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>3.157797</td>
      <td>0.357077</td>
      <td>2.51</td>
      <td>2.87</td>
      <td>3.17</td>
      <td>3.42</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>2.785352</td>
      <td>0.496573</td>
      <td>1.59</td>
      <td>2.44</td>
      <td>2.83</td>
      <td>3.16</td>
      <td>3.69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>1.683542</td>
      <td>0.272111</td>
      <td>1.27</td>
      <td>1.51</td>
      <td>1.66</td>
      <td>1.82</td>
      <td>2.47</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> OD280 </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

##                       
                      Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            3.157797                                 2.785352                                 1.683542(significant variation)
#Minimum                  2.51                                      1.59                                      1.27     (significant variation)
#Max                      4.00                                      3.69                                      2.47     (significant variation)  


```python
#Lets see individual correlation of OD280 with our label Cultivar_Type
Model_DF[['Cultivar_Type','OD280']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>OD280</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.00000</td>
      <td>-0.78823</td>
    </tr>
    <tr>
      <th>OD280</th>
      <td>-0.78823</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="OD280",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a2557c3d0>




![png](output_135_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to OD280 due to which the feature could be significant in differentiating between the CultivarTypes
```


```python
#Considering statistics of Proline we can understand the following details 
Model_DF.groupby('Cultivar_Type')['Proline'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59.0</td>
      <td>1115.711864</td>
      <td>221.520767</td>
      <td>680.0</td>
      <td>987.5</td>
      <td>1095.0</td>
      <td>1280.0</td>
      <td>1680.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71.0</td>
      <td>519.507042</td>
      <td>157.211220</td>
      <td>278.0</td>
      <td>406.5</td>
      <td>495.0</td>
      <td>625.0</td>
      <td>985.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>629.895833</td>
      <td>115.097043</td>
      <td>415.0</td>
      <td>545.0</td>
      <td>627.5</td>
      <td>695.0</td>
      <td>880.0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'> Proline </font>
## <font color='orange'> Cultivar1 </font>   <font color='orange'> Cultivar2 </font>     <font color='orange'> Cultivar3 </font> 

## 
                        Cultivar1                                 Cultivar2                                  Cultivar3  
#No.of values             59                                         71                                         48
#Average value            1115.711864                               519.507042                                629.895833(significant variation)
#Minimum                  680.0                                     278.0                                    415.0     (significant variation)
#Max                      1680.0                                    985.0                                   880.0     (significant variation)  


```python
#Lets see individual correlation of Proline with our label Cultivar_Type
Model_DF[['Cultivar_Type','Proline']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cultivar_Type</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cultivar_Type</th>
      <td>1.000000</td>
      <td>-0.633717</td>
    </tr>
    <tr>
      <th>Proline</th>
      <td>-0.633717</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Visualizing the spread
sns.catplot(x="Cultivar_Type", y="Proline",data=Model_DF)
```




    <seaborn.axisgrid.FacetGrid at 0x1a254d2490>




![png](output_141_1.png)



```python
#The three types are having different spread of datapoints(highest value,lowest values) related to Proline due to which the feature could be significant in differentiating between the CultivarTypes
```


```python

```


```python
#sns.catplot(x="Cultivar_Type", y="Alcohol", kind="box",data=Model_DF.sort_values("Cultivar_Type"))
```


```python
#sns.catplot(x="Cultivar_Type", y="Alcohol", hue="Cultivar_Type", kind="box", data=Model_DF);
```


```python
#Checking the correlation of each feature with target variable by observing the last column in heat map
```


```python
Selected_features = ['Alcohol','MalicAcid','Ash','Alkalinity','Magnesium','Phenols','Flavanoids','NonFlavanoids','Pcyanins','ColorIntensity','Hue','OD280','Proline','Cultivar_Type']
X = Model_DF[Selected_features]

plt.subplots(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()
```


![png](output_147_0.png)


## <font color='purple'>We can see that six features are highly correlated above positive (0.5) - 1 and above negative (0.5) -5 </font>


```python
#On the basis of individual correlation coefficients, we are determining which independent variables are useful in predicting the target value 
#Correlation coefficient value ranges from -1 to +1; closer to 1, stronger the relationship. 
#Also, only correlation coefficients greater than 0.5 in magnitude are considered for further inclusion in the model.
#These variables are considered relevant attributes for prediction of Cultivar Type.
```


```python
#For more clear values lets see correlation matrix instead of heatmap
```


```python
Model_DF.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>MalicAcid</th>
      <th>Ash</th>
      <th>Alkalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>NonFlavanoids</th>
      <th>Pcyanins</th>
      <th>ColorIntensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Cultivar_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Alcohol</th>
      <td>1.000000</td>
      <td>0.094397</td>
      <td>0.211545</td>
      <td>-0.310235</td>
      <td>0.270798</td>
      <td>0.289101</td>
      <td>0.236815</td>
      <td>-0.155929</td>
      <td>0.136698</td>
      <td>0.546364</td>
      <td>-0.071747</td>
      <td>0.072343</td>
      <td>0.643720</td>
      <td>-0.328222</td>
    </tr>
    <tr>
      <th>MalicAcid</th>
      <td>0.094397</td>
      <td>1.000000</td>
      <td>0.164045</td>
      <td>0.288500</td>
      <td>-0.054575</td>
      <td>-0.335167</td>
      <td>-0.411007</td>
      <td>0.292977</td>
      <td>-0.220746</td>
      <td>0.248985</td>
      <td>-0.561296</td>
      <td>-0.368710</td>
      <td>-0.192011</td>
      <td>0.437776</td>
    </tr>
    <tr>
      <th>Ash</th>
      <td>0.211545</td>
      <td>0.164045</td>
      <td>1.000000</td>
      <td>0.443367</td>
      <td>0.286587</td>
      <td>0.128980</td>
      <td>0.115077</td>
      <td>0.186230</td>
      <td>0.009652</td>
      <td>0.258887</td>
      <td>-0.074667</td>
      <td>0.003911</td>
      <td>0.223626</td>
      <td>-0.049643</td>
    </tr>
    <tr>
      <th>Alkalinity</th>
      <td>-0.310235</td>
      <td>0.288500</td>
      <td>0.443367</td>
      <td>1.000000</td>
      <td>-0.083333</td>
      <td>-0.321113</td>
      <td>-0.351370</td>
      <td>0.361922</td>
      <td>-0.197327</td>
      <td>0.018732</td>
      <td>-0.273955</td>
      <td>-0.276769</td>
      <td>-0.440597</td>
      <td>0.517859</td>
    </tr>
    <tr>
      <th>Magnesium</th>
      <td>0.270798</td>
      <td>-0.054575</td>
      <td>0.286587</td>
      <td>-0.083333</td>
      <td>1.000000</td>
      <td>0.214401</td>
      <td>0.195784</td>
      <td>-0.256294</td>
      <td>0.236441</td>
      <td>0.199950</td>
      <td>0.055398</td>
      <td>0.066004</td>
      <td>0.393351</td>
      <td>-0.209179</td>
    </tr>
    <tr>
      <th>Phenols</th>
      <td>0.289101</td>
      <td>-0.335167</td>
      <td>0.128980</td>
      <td>-0.321113</td>
      <td>0.214401</td>
      <td>1.000000</td>
      <td>0.864564</td>
      <td>-0.449935</td>
      <td>0.612413</td>
      <td>-0.055136</td>
      <td>0.433681</td>
      <td>0.699949</td>
      <td>0.498115</td>
      <td>-0.719163</td>
    </tr>
    <tr>
      <th>Flavanoids</th>
      <td>0.236815</td>
      <td>-0.411007</td>
      <td>0.115077</td>
      <td>-0.351370</td>
      <td>0.195784</td>
      <td>0.864564</td>
      <td>1.000000</td>
      <td>-0.537900</td>
      <td>0.652692</td>
      <td>-0.172379</td>
      <td>0.543479</td>
      <td>0.787194</td>
      <td>0.494193</td>
      <td>-0.847498</td>
    </tr>
    <tr>
      <th>NonFlavanoids</th>
      <td>-0.155929</td>
      <td>0.292977</td>
      <td>0.186230</td>
      <td>0.361922</td>
      <td>-0.256294</td>
      <td>-0.449935</td>
      <td>-0.537900</td>
      <td>1.000000</td>
      <td>-0.365845</td>
      <td>0.139057</td>
      <td>-0.262640</td>
      <td>-0.503270</td>
      <td>-0.311385</td>
      <td>0.489109</td>
    </tr>
    <tr>
      <th>Pcyanins</th>
      <td>0.136698</td>
      <td>-0.220746</td>
      <td>0.009652</td>
      <td>-0.197327</td>
      <td>0.236441</td>
      <td>0.612413</td>
      <td>0.652692</td>
      <td>-0.365845</td>
      <td>1.000000</td>
      <td>-0.025250</td>
      <td>0.295544</td>
      <td>0.519067</td>
      <td>0.330417</td>
      <td>-0.499130</td>
    </tr>
    <tr>
      <th>ColorIntensity</th>
      <td>0.546364</td>
      <td>0.248985</td>
      <td>0.258887</td>
      <td>0.018732</td>
      <td>0.199950</td>
      <td>-0.055136</td>
      <td>-0.172379</td>
      <td>0.139057</td>
      <td>-0.025250</td>
      <td>1.000000</td>
      <td>-0.521813</td>
      <td>-0.428815</td>
      <td>0.316100</td>
      <td>0.265668</td>
    </tr>
    <tr>
      <th>Hue</th>
      <td>-0.071747</td>
      <td>-0.561296</td>
      <td>-0.074667</td>
      <td>-0.273955</td>
      <td>0.055398</td>
      <td>0.433681</td>
      <td>0.543479</td>
      <td>-0.262640</td>
      <td>0.295544</td>
      <td>-0.521813</td>
      <td>1.000000</td>
      <td>0.565468</td>
      <td>0.236183</td>
      <td>-0.617369</td>
    </tr>
    <tr>
      <th>OD280</th>
      <td>0.072343</td>
      <td>-0.368710</td>
      <td>0.003911</td>
      <td>-0.276769</td>
      <td>0.066004</td>
      <td>0.699949</td>
      <td>0.787194</td>
      <td>-0.503270</td>
      <td>0.519067</td>
      <td>-0.428815</td>
      <td>0.565468</td>
      <td>1.000000</td>
      <td>0.312761</td>
      <td>-0.788230</td>
    </tr>
    <tr>
      <th>Proline</th>
      <td>0.643720</td>
      <td>-0.192011</td>
      <td>0.223626</td>
      <td>-0.440597</td>
      <td>0.393351</td>
      <td>0.498115</td>
      <td>0.494193</td>
      <td>-0.311385</td>
      <td>0.330417</td>
      <td>0.316100</td>
      <td>0.236183</td>
      <td>0.312761</td>
      <td>1.000000</td>
      <td>-0.633717</td>
    </tr>
    <tr>
      <th>Cultivar_Type</th>
      <td>-0.328222</td>
      <td>0.437776</td>
      <td>-0.049643</td>
      <td>0.517859</td>
      <td>-0.209179</td>
      <td>-0.719163</td>
      <td>-0.847498</td>
      <td>0.489109</td>
      <td>-0.499130</td>
      <td>0.265668</td>
      <td>-0.617369</td>
      <td>-0.788230</td>
      <td>-0.633717</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'>Flavonoids is the first high negatively correlated value: -0.847498  </font>
## <font color='purple'>OD280 is the second high negatively correlated value: -0.788230  </font>
## <font color='purple'>Phenols is the third high negatively correlated value: -0.719163 </font>
## <font color='purple'>Hue is the fourth high negatively correlated value: -0.617369  </font>
## <font color='purple'>Proline is the fifth high negatively correlated value: -0.633717  </font>
## <font color='purple'>Alkalinity  is the sixth positively correlated value: 0.517859  </font>



```python
#Checking the distribution again
Model_DF.groupby(['Cultivar_Type']).size()
```




    Cultivar_Type
    0    59
    1    71
    2    48
    dtype: int64




```python
Model_DF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>MalicAcid</th>
      <th>Ash</th>
      <th>Alkalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>NonFlavanoids</th>
      <th>Pcyanins</th>
      <th>ColorIntensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
      <th>Cultivar_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='purple'>Considering only highly correlated features Flavanoids,OD280,Phenols,Hue,Proline,Alkalinity for training and testing the model </font>


```python
Wine_Corr_features_DF = Model_DF[['Alkalinity','Phenols','Flavanoids','Hue','OD280','Proline']]
```


```python
#Input data to model (features)
Wine_Corr_features_DF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alkalinity</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.6</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.2</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.6</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.8</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
Wine_Target_DF = Model_DF['Cultivar_Type']
```


```python
#Label data to model (target variable)
Wine_Target_DF.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: Cultivar_Type, dtype: int64



## Question5.	If your chosen algorithm demands training and test datasets, split your wine dataset using an 80/20 split. If dataset is split, evaluate your training and test datasets to ensure they are representative of your full data set. 

## <font color='purple'>To Train and Test the Logistic Regression Model, split dataset 80-20%</font>


```python
#Importing the train test split function from sklearn
from sklearn.model_selection import train_test_split
```


```python
#Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(Wine_Corr_features_DF,Wine_Target_DF, test_size=0.20, random_state = 1)
```


```python
#Loking at train data
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alkalinity</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>91</th>
      <td>22.0</td>
      <td>1.45</td>
      <td>1.25</td>
      <td>1.05</td>
      <td>2.65</td>
      <td>450</td>
    </tr>
    <tr>
      <th>81</th>
      <td>18.8</td>
      <td>2.20</td>
      <td>2.53</td>
      <td>1.16</td>
      <td>3.14</td>
      <td>714</td>
    </tr>
    <tr>
      <th>114</th>
      <td>22.5</td>
      <td>2.56</td>
      <td>2.29</td>
      <td>0.93</td>
      <td>3.19</td>
      <td>385</td>
    </tr>
    <tr>
      <th>48</th>
      <td>18.8</td>
      <td>2.75</td>
      <td>2.92</td>
      <td>1.07</td>
      <td>2.75</td>
      <td>1060</td>
    </tr>
    <tr>
      <th>54</th>
      <td>16.4</td>
      <td>2.60</td>
      <td>2.90</td>
      <td>0.92</td>
      <td>3.20</td>
      <td>1060</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>133</th>
      <td>21.5</td>
      <td>1.70</td>
      <td>1.20</td>
      <td>0.78</td>
      <td>1.29</td>
      <td>600</td>
    </tr>
    <tr>
      <th>137</th>
      <td>25.0</td>
      <td>1.79</td>
      <td>0.60</td>
      <td>0.82</td>
      <td>1.69</td>
      <td>515</td>
    </tr>
    <tr>
      <th>72</th>
      <td>24.0</td>
      <td>1.88</td>
      <td>1.84</td>
      <td>0.98</td>
      <td>2.78</td>
      <td>472</td>
    </tr>
    <tr>
      <th>140</th>
      <td>21.0</td>
      <td>1.54</td>
      <td>0.50</td>
      <td>0.77</td>
      <td>2.31</td>
      <td>600</td>
    </tr>
    <tr>
      <th>37</th>
      <td>18.0</td>
      <td>2.45</td>
      <td>2.43</td>
      <td>1.12</td>
      <td>2.51</td>
      <td>1105</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 6 columns</p>
</div>




```python
#Checking the shape of all splits
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
```




    ((142, 6), (142,), (36, 6), (36,))




```python
# Get a tuple of unique values & their frequency in numpy array for trainig data
uniqueValues, occurCount = np.unique(Y_train, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)
```

    Unique Values :  [0 1 2]
    Occurrence Count :  [45 58 39]



```python
#Total records split for training is 142
#Number of Cultivar1 type in training data are (45) -- (45/142)-- 31.69%
#Number of Cultivar2 type in training data are (58) -- (58/142)-- 40.84% 
#Number of Cultivar3 type in training data are (39) -- (39/142)-- 27.46%
################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (71) -- (71/115)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%
```


```python
#displaying training datasets
print(X_train,Y_train)
```

         Alkalinity  Phenols  Flavanoids   Hue  OD280  Proline
    91         22.0     1.45        1.25  1.05   2.65      450
    81         18.8     2.20        2.53  1.16   3.14      714
    114        22.5     2.56        2.29  0.93   3.19      385
    48         18.8     2.75        2.92  1.07   2.75     1060
    54         16.4     2.60        2.90  0.92   3.20     1060
    ..          ...      ...         ...   ...    ...      ...
    133        21.5     1.70        1.20  0.78   1.29      600
    137        25.0     1.79        0.60  0.82   1.69      515
    72         24.0     1.88        1.84  0.98   2.78      472
    140        21.0     1.54        0.50  0.77   2.31      600
    37         18.0     2.45        2.43  1.12   2.51     1105
    
    [142 rows x 6 columns] 91     1
    81     1
    114    1
    48     0
    54     0
          ..
    133    2
    137    2
    72     1
    140    2
    37     0
    Name: Cultivar_Type, Length: 142, dtype: int64



```python
# Get a tuple of unique values & their frequency in numpy array for testing data
uniqueValues, occurCount = np.unique(Y_test, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)
```

    Unique Values :  [0 1 2]
    Occurrence Count :  [14 13  9]



```python
#Total records split for testing is 36
#Number of Cultivar1 type in training data are (14) -- (14/36)-- 38.88%
#Number of Cultivar2 type in training data are (13) -- (13/36)-- 36.11% 
#Number of Cultivar3 type in training data are (9) -- (9/36)-- 25.00%
################################################################################
#Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%
#Number of Cultivar2 type in training data are (71) -- (71/178)-- 39.88% 
#Number of Cultivar3 type in training data are (48) -- (48/178)-- 26.96%
```

## <font color='orange'>Total records split for training is 142</font>

### <font color='purple'>Number of Cultivar1 type in training data are (45) -- (45/142)-- 31.69%</font>
### <font color='purple'>Number of Cultivar2 type in training data are (58) -- (58/142)-- 40.84%</font>
### <font color='purple'>Number of Cultivar3 type in training data are (39) -- (39/142)-- 27.46%</font>

## <font color='orange'>Total records split for testing is 36</font>
### <font color='purple'>Number of Cultivar1 type in training data are (14) -- (14/36)-- 38.88%</font>
### <font color='purple'>Number of Cultivar2 type in training data are (58) -- (13/36)-- 36.11%</font>
### <font color='purple'>Number of Cultivar3 type in training data are (39) -- (9/136)-- 25.00%</font>
        
## <font color='orange'> Overall records are 178 </font>      
### <font color='purple'>Number of Cultivar1 type in training data are (59) -- (59/178)-- 33.14%</font>
### <font color='purple'>Number of Cultivar2 type in training data are (71) -- (71/115)-- 39.88%</font>
### <font color='purple'>Number of Cultivar3 type in training data are (48) -- (48/115)-- 26.96%</font>

# <font color='blue'>The training dataset split is representative of overall data as there is only 1 or 2 percents split variation compared to percentages of overall data</font> 
# <font color='blue'>The testing dataset split is also slightly representative of overall data as there is 4 or 5 percents split variation compared to percentages of overall data</font>


```python
#Import Logistic regression from SKLearn Libraries
from sklearn.linear_model import LogisticRegression
```

# Question6.Use the relevant portion of your dataset to train the model of your selected machine learning algorithm. Do all the necessary preprocessing to determine the parameters for your selected algorithm. For example, you will need to specify (and justify) the number of clusters if you choose to use KMeans clustering algorithm via the Elbow curve, Silhouette analysis, etc. 


```python
#Creting an instance of Logistic regression using multi_class function
log_reg = LogisticRegression(solver='newton-cg',multi_class='multinomial')
#Applying training data to the model
log_reg.fit(X_train,Y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='multinomial', n_jobs=None, penalty='l2',
                       random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                       warm_start=False)




```python
#Generating training predictions
```


```python
#Generate predictions to evaluate the trained model using X_Train data. 
Ytrain_predict = log_reg.predict(X_train)
```


```python
# Our resulting Y_predict variable is of shape (142,). so, it needs to be converted to (142,1) 2-D array
# to a new result dataframe
predict_ytrain = Ytrain_predict.reshape(-1,1)
print(predict_ytrain.shape)
```

    (142, 1)



```python
# The Y_train contains values of our target variable (Cultivar). There were 142 records in the training data. So, Y_train
# has a shape of (142,) - one dimesional. We'll need to reshape this into 2-D (142 rows, 1 column of Y-values)

train_y = (Y_train.values).reshape(-1,1)
print(train_y.shape)
print(train_y.size)
```

    (142, 1)
    142



```python
# We need to obtain probabilities for our predictions. For this, we need to use predict_proba() function of the 
# logistic regression model we instantiated earlier during model training and predictions. 

train_predicted_probs = log_reg.predict_proba(X_train)
print(train_predicted_probs)
```

    [[4.78513363e-04 8.52889144e-01 1.46632343e-01]
     [2.01475016e-01 7.97550612e-01 9.74371886e-04]
     [8.39946953e-04 9.94999122e-01 4.16093110e-03]
     [9.82223215e-01 1.77245754e-02 5.22093576e-05]
     [9.94474029e-01 5.52131105e-03 4.66014894e-06]
     [3.87354925e-03 8.94642696e-01 1.01483755e-01]
     [2.14767516e-05 4.67127279e-02 9.53265795e-01]
     [8.42262032e-01 1.57731415e-01 6.55252530e-06]
     [9.81741971e-01 1.82520413e-02 5.98756572e-06]
     [9.32995227e-01 6.69810230e-02 2.37500386e-05]
     [2.17649359e-01 7.72067397e-01 1.02832445e-02]
     [9.98514264e-01 1.47974327e-03 5.99264328e-06]
     [9.99999617e-01 3.82804368e-07 1.35185106e-10]
     [9.99862524e-01 1.37417040e-04 5.88200682e-08]
     [2.88137109e-04 8.36745508e-01 1.62966355e-01]
     [8.40188355e-04 8.66183081e-02 9.12541504e-01]
     [3.14111133e-04 4.93488797e-02 9.50337009e-01]
     [9.99814702e-01 1.84841430e-04 4.56626842e-07]
     [1.31696931e-03 9.45845040e-01 5.28379911e-02]
     [7.87916757e-01 2.09429676e-01 2.65356700e-03]
     [8.94594675e-01 1.05365618e-01 3.97067562e-05]
     [3.80255255e-03 9.95981890e-01 2.15557808e-04]
     [1.85378577e-03 1.12243467e-01 8.85902748e-01]
     [3.45558323e-02 9.65417143e-01 2.70251365e-05]
     [1.47071633e-03 9.96418089e-01 2.11119499e-03]
     [9.92805585e-01 7.18108595e-03 1.33287633e-05]
     [1.42202319e-04 5.26165245e-02 9.47241273e-01]
     [1.35929861e-02 9.31343287e-01 5.50637270e-02]
     [1.40713212e-04 1.33595311e-01 8.66263976e-01]
     [9.99279284e-01 7.20696148e-04 1.95638251e-08]
     [8.76246891e-03 9.87381385e-01 3.85614571e-03]
     [9.93484717e-01 6.49353036e-03 2.17528295e-05]
     [3.91131701e-04 9.99231049e-01 3.77819102e-04]
     [8.46735172e-01 1.53117715e-01 1.47112746e-04]
     [3.81937358e-05 8.15274982e-02 9.18434308e-01]
     [4.25533083e-04 6.48563280e-02 9.34718139e-01]
     [5.35881083e-04 3.29801814e-02 9.66483938e-01]
     [4.68586115e-03 3.58052992e-02 9.59508840e-01]
     [4.63255364e-04 9.06117396e-01 9.34193491e-02]
     [7.54331467e-04 9.84061331e-01 1.51843376e-02]
     [5.13959322e-01 4.85478819e-01 5.61858797e-04]
     [1.49955590e-03 5.19928954e-02 9.46507549e-01]
     [9.98841241e-01 1.15777006e-03 9.89273616e-07]
     [3.79405081e-02 1.67389944e-01 7.94669548e-01]
     [2.38903270e-03 1.49411026e-01 8.48199941e-01]
     [9.99895891e-01 1.04006899e-04 1.02207896e-07]
     [2.91137858e-03 9.95511661e-01 1.57696004e-03]
     [9.97951911e-01 2.04792472e-03 1.63784331e-07]
     [1.45553659e-04 1.66213629e-02 9.83233083e-01]
     [2.70690930e-03 9.37488700e-01 5.98043906e-02]
     [6.42512894e-02 9.33144848e-01 2.60386256e-03]
     [9.92852768e-01 7.10268728e-03 4.45449566e-05]
     [1.68800799e-03 9.98216181e-01 9.58108237e-05]
     [3.28884734e-02 8.94801013e-01 7.23105139e-02]
     [3.38221332e-03 9.92234944e-01 4.38284309e-03]
     [9.96538381e-01 3.46107685e-03 5.42566464e-07]
     [1.94494212e-02 9.65381527e-01 1.51690515e-02]
     [6.92498759e-01 3.03168270e-01 4.33297037e-03]
     [7.88775051e-04 7.80230327e-01 2.18980898e-01]
     [5.88198174e-03 3.96611009e-02 9.54456917e-01]
     [4.49073094e-03 9.79536585e-01 1.59726838e-02]
     [1.46690454e-02 9.68402942e-01 1.69280127e-02]
     [1.61664079e-03 9.97443050e-01 9.40309661e-04]
     [2.52835604e-02 9.71474367e-01 3.24207289e-03]
     [9.99991913e-01 8.08010537e-06 7.28319617e-09]
     [9.86852191e-01 1.30459493e-02 1.01859384e-04]
     [9.52203544e-01 4.77255658e-02 7.08903150e-05]
     [2.31039918e-03 2.30449706e-02 9.74644630e-01]
     [9.74489653e-01 2.54683573e-02 4.19897276e-05]
     [7.63027918e-04 6.50309518e-01 3.48927454e-01]
     [2.61410102e-04 4.55736636e-02 9.54164926e-01]
     [2.06037475e-05 2.71178513e-02 9.72861545e-01]
     [9.99412875e-01 5.73000358e-04 1.41244918e-05]
     [1.44386955e-01 8.53767457e-01 1.84558747e-03]
     [9.83108459e-01 1.68671453e-02 2.43956569e-05]
     [6.49824826e-04 9.63617532e-01 3.57326433e-02]
     [9.17377123e-03 9.88854821e-01 1.97140784e-03]
     [9.95900535e-01 4.09888153e-03 5.83510032e-07]
     [5.75265547e-05 1.34398205e-01 8.65544268e-01]
     [1.01152326e-01 8.93989510e-01 4.85816332e-03]
     [2.18880716e-04 9.71413896e-01 2.83672233e-02]
     [3.31513014e-05 4.41999075e-02 9.55766941e-01]
     [1.82624728e-03 1.14360751e-01 8.83813001e-01]
     [2.13458703e-03 9.43787999e-01 5.40774142e-02]
     [1.81562671e-01 8.18372343e-01 6.49855267e-05]
     [2.91606141e-04 9.94582688e-01 5.12570585e-03]
     [1.52535722e-03 1.08696860e-01 8.89777783e-01]
     [1.02616363e-03 1.32396876e-01 8.66576960e-01]
     [1.22463342e-04 9.89634920e-01 1.02426165e-02]
     [9.99697816e-01 3.01643652e-04 5.40821406e-07]
     [5.40798228e-03 8.97400802e-01 9.71912160e-02]
     [9.80848837e-01 1.90589658e-02 9.21976410e-05]
     [7.44888736e-04 4.03944911e-01 5.95310200e-01]
     [1.88468327e-01 8.10985063e-01 5.46610731e-04]
     [1.59198020e-04 9.99421613e-01 4.19188874e-04]
     [9.99749140e-01 2.50850913e-04 8.79578607e-09]
     [9.99401811e-01 5.97621089e-04 5.68210497e-07]
     [9.37073482e-04 9.96837354e-01 2.22557227e-03]
     [2.22079118e-01 7.76825292e-01 1.09559048e-03]
     [7.85246790e-01 2.14618897e-01 1.34313748e-04]
     [9.99996213e-01 3.78568946e-06 1.57901768e-09]
     [2.41116272e-04 1.65027001e-01 8.34731882e-01]
     [9.99724186e-01 2.75730958e-04 8.26979230e-08]
     [3.12441257e-04 2.77287861e-01 7.22399698e-01]
     [1.93004937e-04 6.22555017e-02 9.37551493e-01]
     [9.98886256e-01 1.10618972e-03 7.55408122e-06]
     [1.10133816e-02 9.88976963e-01 9.65492235e-06]
     [3.24547231e-04 9.98287729e-01 1.38772400e-03]
     [2.70624384e-04 2.00837335e-02 9.79645642e-01]
     [9.92667069e-01 7.33017087e-03 2.75976611e-06]
     [1.28879712e-02 7.89954329e-01 1.97157700e-01]
     [2.10699635e-04 9.87738819e-01 1.20504816e-02]
     [9.97651894e-01 2.34786847e-03 2.37978992e-07]
     [9.99676248e-01 3.23423654e-04 3.28132745e-07]
     [9.95389443e-01 4.60998190e-03 5.74707854e-07]
     [5.46768379e-04 8.67466791e-01 1.31986440e-01]
     [2.57051173e-03 9.97294465e-01 1.35023476e-04]
     [9.99810347e-01 1.89172503e-04 4.80904173e-07]
     [4.48601614e-04 2.65880864e-02 9.72963312e-01]
     [1.76623530e-02 1.55486911e-01 8.26850736e-01]
     [5.41615114e-04 9.20539741e-01 7.89186441e-02]
     [2.63644216e-03 4.43566519e-01 5.53797039e-01]
     [6.43614785e-02 7.57816949e-01 1.77821573e-01]
     [9.99173103e-01 8.26800304e-04 9.69081682e-08]
     [4.19047989e-05 8.85231613e-02 9.11434934e-01]
     [2.50544093e-04 3.96802858e-03 9.95781427e-01]
     [1.26239404e-04 1.79800961e-01 8.20072799e-01]
     [1.53649295e-04 1.01341348e-01 8.98505003e-01]
     [3.26907439e-06 2.94087365e-02 9.70587994e-01]
     [7.89503516e-03 9.49334600e-01 4.27703651e-02]
     [8.07262307e-01 1.92730607e-01 7.08582552e-06]
     [3.49280200e-01 6.45234011e-01 5.48578843e-03]
     [1.20989766e-03 1.40664315e-01 8.58125787e-01]
     [6.08442201e-04 9.98510239e-01 8.81318436e-04]
     [7.11965566e-03 8.79931269e-01 1.12949075e-01]
     [5.77637648e-02 1.48841830e-01 7.93394405e-01]
     [2.04340209e-03 9.96082238e-01 1.87436006e-03]
     [2.83205727e-04 1.40533128e-01 8.59183666e-01]
     [7.81372021e-06 3.12759589e-02 9.68716227e-01]
     [9.61136996e-04 9.37593781e-01 6.14450822e-02]
     [7.23899181e-04 1.71033156e-01 8.28242945e-01]
     [9.84914431e-01 1.48799218e-02 2.05647350e-04]]


# As mentioned the output of the multicalss logistic regression is set of probabilities based on which the predictions are generated


```python
#Finally, we add all five variables into a Pandas Dataframe for display purposes. 

np.set_printoptions(suppress=True)  # this is to prevent small values being displayed in scientific notations

train_prob_results_df = pd.DataFrame(train_predicted_probs)
train_prob_results_df["Predicted"] = predict_ytrain
train_prob_results_df["Actual"] = train_y
train_prob_results_df.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>Predicted</th>
      <th>Actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000479</td>
      <td>0.852889</td>
      <td>0.146632</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.201475</td>
      <td>0.797551</td>
      <td>0.000974</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000840</td>
      <td>0.994999</td>
      <td>0.004161</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.982223</td>
      <td>0.017725</td>
      <td>0.000052</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.994474</td>
      <td>0.005521</td>
      <td>0.000005</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.003874</td>
      <td>0.894643</td>
      <td>0.101484</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000021</td>
      <td>0.046713</td>
      <td>0.953266</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.842262</td>
      <td>0.157731</td>
      <td>0.000007</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.981742</td>
      <td>0.018252</td>
      <td>0.000006</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.932995</td>
      <td>0.066981</td>
      <td>0.000024</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# 7.Using appropriate metrics for your chosen algorithm, evaluate the trained model. Explain and justify the worthiness of your trained model. 


```python
# Let's evaluate the trained model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(train_y,predict_ytrain))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.96      0.96        45
               1       0.96      0.93      0.95        58
               2       0.95      1.00      0.97        39
    
        accuracy                           0.96       142
       macro avg       0.96      0.96      0.96       142
    weighted avg       0.96      0.96      0.96       142
    



```python
conf_matrix = metrics.confusion_matrix(train_y,predict_ytrain)
conf_matrix
```




    array([[43,  2,  0],
           [ 2, 54,  2],
           [ 0,  0, 39]])




```python
print("Accuracy:",metrics.accuracy_score(train_y,predict_ytrain))
```

    Accuracy: 0.9577464788732394



```python
print("Precision:",metrics.precision_score(train_y,predict_ytrain, average = "macro"))
print("Recall:",metrics.recall_score(train_y,predict_ytrain,average = "macro"))
```

    Precision: 0.9570202606787973
    Recall: 0.9621966794380588



```python
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
```

    Area under curve :  0.9704574136792443


# The model is trained well as the AOC is almost 1.00(0.97) and all other metrics of Accuracy, Recall, Precision,,and F1-score are all above 95% and close 1.0, the highest value for all; metrics. 
# This means that the trained model couldidentify Cultivar0 as Cultivar0, Cultivar1 as Cultivar1 and  Cultivar2 as Cultivar2 . The model is trained to be able to distinguish between (and therfore, predict correctly) between Cultivar_Types.

# 8.	Next, use the relevant portion of your dataset (as dictated by the chosen algorithm) to evaluate the performance of your model. Again, use all relevant metrics for your algorithm to discuss the outcome in terms of model’s accuracy and usefulness in generating predictions. These may include such metrics as SSE, MSSE, Silhouette scores, completeness scores, confusion matrix, AOC curve, etc. as dictated by and available for your chosen machine language algorithm.


```python
#Generate predictions to evaluate the testing model using X_Test data and predicting Y

Ytest_predict  = log_reg.predict(X_test)
```


```python
# Our resulting Y_predict variable is of shape (36,). so, it needs to be converted to (36,1) 2-D array
# to a new result dataframe
predict_ytest = Ytest_predict.reshape(-1,1)
print(predict_ytest.shape)
```

    (36, 1)



```python
# The Y_test contains values of our target variable (Cultivar). There were 36 records in the test data. So, Y_test
# has a shape of (36,) - one dimesional. We'll need to reshape this into 2-D (36 rows, 1 column of Y-values)

test_y = (Y_test.values).reshape(-1,1)
print(test_y.shape)
print(test_y.size)
```

    (36, 1)
    36



```python
# We need to obtain probabilities for our predictions. For this, we need to use predict_proba() function of the 
# logistic regression model we instantiated earlier during model training and predictions. 

test_predicted_probs = log_reg.predict_proba(X_test)
print(test_predicted_probs)
```

    [[0.00124438 0.14541987 0.85333574]
     [0.00024931 0.99177917 0.00797152]
     [0.90856123 0.09142627 0.0000125 ]
     [0.11310765 0.87362845 0.0132639 ]
     [0.99991003 0.00008965 0.00000032]
     [0.00030199 0.1318604  0.86783761]
     [0.01118626 0.97384264 0.01497109]
     [0.99999882 0.00000118 0.        ]
     [0.00012242 0.12428351 0.87559407]
     [0.00112898 0.87612219 0.12274883]
     [0.99941177 0.00058347 0.00000477]
     [0.19064413 0.80681374 0.00254213]
     [0.00041863 0.99630261 0.00327876]
     [0.99905849 0.00094147 0.00000005]
     [0.01729748 0.97969397 0.00300855]
     [0.00059132 0.97758738 0.0218213 ]
     [0.00000541 0.0668728  0.93312179]
     [0.91337946 0.08655103 0.00006951]
     [0.0002741  0.98940869 0.0103172 ]
     [0.99998852 0.00001142 0.00000005]
     [0.9996804  0.00031941 0.00000019]
     [0.00107912 0.99799582 0.00092506]
     [0.00237723 0.33865726 0.65896551]
     [0.73469882 0.26158409 0.00371709]
     [0.82340719 0.17657951 0.0000133 ]
     [0.00025132 0.15485769 0.84489099]
     [0.99025488 0.00974397 0.00000115]
     [0.99138541 0.00860528 0.0000093 ]
     [0.99929027 0.0007056  0.00000414]
     [0.00030596 0.0584062  0.94128783]
     [0.00294344 0.99669603 0.00036052]
     [0.00066902 0.13741933 0.86191165]
     [0.0000976  0.17098077 0.82892163]
     [0.99998808 0.00001191 0.00000001]
     [0.1142277  0.88500333 0.00076897]
     [0.00063703 0.99863766 0.00072531]]



```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>Predicted</th>
      <th>Actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00124438217</td>
      <td>0.14541987367</td>
      <td>0.85333574416</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00024930729</td>
      <td>0.99177917166</td>
      <td>0.00797152105</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.90856123347</td>
      <td>0.09142626903</td>
      <td>0.00001249749</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11310764941</td>
      <td>0.87362844791</td>
      <td>0.01326390267</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.99991003185</td>
      <td>0.00008964675</td>
      <td>0.00000032141</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00030199243</td>
      <td>0.13186040241</td>
      <td>0.86783760516</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.01118626377</td>
      <td>0.97384264313</td>
      <td>0.01497109310</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99999882476</td>
      <td>0.00000117515</td>
      <td>0.00000000010</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.00012242006</td>
      <td>0.12428350881</td>
      <td>0.87559407113</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.00112897634</td>
      <td>0.87612219227</td>
      <td>0.12274883139</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.99941176716</td>
      <td>0.00058346551</td>
      <td>0.00000476734</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.19064412846</td>
      <td>0.80681374299</td>
      <td>0.00254212855</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.00041863268</td>
      <td>0.99630260756</td>
      <td>0.00327875976</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.99905848721</td>
      <td>0.00094146509</td>
      <td>0.00000004770</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01729747896</td>
      <td>0.97969396851</td>
      <td>0.00300855253</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.00059131657</td>
      <td>0.97758738230</td>
      <td>0.02182130113</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.00000540635</td>
      <td>0.06687279959</td>
      <td>0.93312179405</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.91337946015</td>
      <td>0.08655102599</td>
      <td>0.00006951385</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.00027410232</td>
      <td>0.98940869350</td>
      <td>0.01031720418</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.99998852246</td>
      <td>0.00001142332</td>
      <td>0.00000005422</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.99968039914</td>
      <td>0.00031941176</td>
      <td>0.00000018909</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.00107912035</td>
      <td>0.99799581552</td>
      <td>0.00092506414</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.00237722967</td>
      <td>0.33865726005</td>
      <td>0.65896551028</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.73469882088</td>
      <td>0.26158408928</td>
      <td>0.00371708985</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.82340719235</td>
      <td>0.17657950738</td>
      <td>0.00001330028</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.00025132112</td>
      <td>0.15485769103</td>
      <td>0.84489098786</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.99025487726</td>
      <td>0.00974397251</td>
      <td>0.00000115023</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.99138541197</td>
      <td>0.00860528314</td>
      <td>0.00000930490</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.99929026581</td>
      <td>0.00070559523</td>
      <td>0.00000413895</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.00030596273</td>
      <td>0.05840620291</td>
      <td>0.94128783436</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.00294344321</td>
      <td>0.99669603354</td>
      <td>0.00036052325</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.00066901841</td>
      <td>0.13741933046</td>
      <td>0.86191165113</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.00009759693</td>
      <td>0.17098077073</td>
      <td>0.82892163233</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.99998807994</td>
      <td>0.00001191226</td>
      <td>0.00000000780</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.11422770141</td>
      <td>0.88500332758</td>
      <td>0.00076897101</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.00063703280</td>
      <td>0.99863765949</td>
      <td>0.00072530772</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#In case of more data we could have trained the model more better
```


```python
# Let's evaluate the model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(Y_test,Ytest_predict))
```

                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93        14
               1       0.92      0.92      0.92        13
               2       1.00      1.00      1.00         9
    
        accuracy                           0.94        36
       macro avg       0.95      0.95      0.95        36
    weighted avg       0.94      0.94      0.94        36
    



```python
conf_matrix = metrics.confusion_matrix(Y_test, Ytest_predict)
conf_matrix
```




    array([[13,  1,  0],
           [ 1, 12,  0],
           [ 0,  0,  9]])



# Understanding the confusion metrics output of multiclass-classification
                     
                       #Predicted values
                             0   1   2
                      0  ([[13,  1,  0],
#Actual values        1    [ 1, 12,  0],
                      2    [ 0,  0,  9]])
                      
Based on the Confusion Matrix, we can see that the predictive performance of the trained model is very
good. Of the total 36 cases, 94.4% are predicted correctly (36.11% Cultivar0, 33.33% Cultivar1 and 25% Cultivar2 ) while only
5.6% are incorrectly predicted. This is further supported by the fact that Recall,Accuracy, Precision, and
F1-score is >= 0.94, very close to 1.0, the ideal score. 


```python
print("Accuracy:",metrics.accuracy_score(Y_test, Ytest_predict))
```

    Accuracy: 0.9444444444444444



```python
print("Precision:",metrics.precision_score(Y_test,Ytest_predict, average = "macro"))
print("Recall:",metrics.recall_score(Y_test,Ytest_predict,average = "macro"))
```

    Precision: 0.9505494505494506
    Recall: 0.9505494505494506



```python
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
```

    Area under curve :  0.9604525908873734


# The AOC is also close to 1.0. Hence, the trained Logistic Regression model is useful in accuaretly predicting the Cultivar type based on the chemical characteristics of wine


            ## ######################################################################
            Accuracy of the model is : 94.4# percent(The model can calculate 94% accurately)
            ######################################################################
            Recall score of the complete model is 95%
            Recall score of predicting Cultivar = 0   is  93%
            Recall score of predicting Cultivar = 1   is  92%
            Recall score of predicting Cultivar = 2   is 100%
            #####################################################################
            Precision score of the complete model is 95%
            Precision score of predicting Cultivar = 0   is  93%
            Precision score of predicting Cultivar = 1   is  92%
            Precision score of predicting Cultivar = 2   is 100%
            #####################################################################
            F1 score of the complete model is 95%
            F1 score of predicting Cultivar = 0   is  93%
            F1 score of predicting Cultivar = 1   is  92%
            F1 score of predicting Cultivar = 2   is 100%
            #####################################################################

        ## #####
        # Area Under the Curve is calculated =0.96
        # Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s and 2's as 2's. By analogy, Higher the AUC, better the model is at distinguishing between Cultivar Types under three to three categories
        #As our Classification, auc score is 0.96 i.e; the model can classify between 0's, 1's,2's and predict them with around 94% accuracy
        #This indicates our model is worthy and also a good model in predicting the survival Cultivar categories

       ## #####################################################################
       #Recall:
        It talks about the quality of the machine learning model when it comes
        to predicting a positive class. So out of total positive classes, how many
        was the model able to predict correctly? This metric is widely used as
        evaluation criteria for classification models.
        The recall values of our model are
        ######################################################################
            Recall score of the complete model is 95%
            Recall score of predicting Cultivar = 0   is  93%
            Recall score of predicting Cultivar = 1   is  92%
            Recall score of predicting Cultivar = 2   is 100%
        #which is good
        #####################################################################
        #Precision:
        Precision is about the number of actual positive cases out of all the positive
        cases predicted by the model
        The precision values of our model are
        #####################################################################
            Precision score of the complete model is 95%
            Precision score of predicting Cultivar = 0   is  93%
            Precision score of predicting Cultivar = 1   is  92%
            Precision score of predicting Cultivar = 2   is 100%
        #which are good
        ################################################################
        #F1 Score:
        It considers both the precision p and the recall r of the test to compute the score
        The F1score values of our model are
        #which are good
        #####################################################################
            F1 score of the complete model is 95%
            F1 score of predicting Cultivar = 0   is  93%
            F1 score of predicting Cultivar = 1   is  92%
            F1 score of predicting Cultivar = 2   is 100%
        #####################################################################
        # Area Under the Curve is calculated =0.9604525908873734
        # Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between Cultivartypes that
        #As our Classification, auc score is 0.96 i.e; the model can classify between 0's, 1's, 2's and predict them with around 94# accuracy
        #This indicates our model is worthy and also a good model in predicting the Cultivar types
        ########################################################################
        Accuracy of the model is :94.4 percent(The model can calculate 94% accurately)

# In case of non linear data and high overlapping between the target classes we can use the following algorithms(Decision Tree and Random Forest) for classification when complexity and time are not a matter of issue(Optional)


```python
# Checking Decision Tree Classifier
```


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
```


```python
#Creating an instance of Decision tree classifier
DTC = DecisionTreeClassifier()

# Apllying Decision Tree Classifer on the training data
DTC = DTC.fit(X_train,Y_train)

DTC_prob = DTC.predict_proba(X_test)
#Predict the response for test dataset
DTC_y_pred = DTC.predict(X_test)
```


```python
#Looking at the predictions
DTC_y_pred
```




    array([1, 1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 1, 1, 2, 0, 1, 0, 0, 1,
           1, 0, 0, 2, 0, 0, 0, 2, 1, 2, 2, 0, 1, 1])




```python
#Creating a pandas dataframe for Decision tree result
DTC_results_df = pd.DataFrame(DTC_prob)
test_prob_results_df["Predicted"] = DTC_y_pred
test_prob_results_df["Actual"] = test_y
test_prob_results_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>Predicted</th>
      <th>Actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00124438217</td>
      <td>0.14541987367</td>
      <td>0.85333574416</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00024930729</td>
      <td>0.99177917166</td>
      <td>0.00797152105</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.90856123347</td>
      <td>0.09142626903</td>
      <td>0.00001249749</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11310764941</td>
      <td>0.87362844791</td>
      <td>0.01326390267</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.99991003185</td>
      <td>0.00008964675</td>
      <td>0.00000032141</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00030199243</td>
      <td>0.13186040241</td>
      <td>0.86783760516</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.01118626377</td>
      <td>0.97384264313</td>
      <td>0.01497109310</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99999882476</td>
      <td>0.00000117515</td>
      <td>0.00000000010</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.00012242006</td>
      <td>0.12428350881</td>
      <td>0.87559407113</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.00112897634</td>
      <td>0.87612219227</td>
      <td>0.12274883139</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.99941176716</td>
      <td>0.00058346551</td>
      <td>0.00000476734</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.19064412846</td>
      <td>0.80681374299</td>
      <td>0.00254212855</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.00041863268</td>
      <td>0.99630260756</td>
      <td>0.00327875976</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.99905848721</td>
      <td>0.00094146509</td>
      <td>0.00000004770</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01729747896</td>
      <td>0.97969396851</td>
      <td>0.00300855253</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.00059131657</td>
      <td>0.97758738230</td>
      <td>0.02182130113</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.00000540635</td>
      <td>0.06687279959</td>
      <td>0.93312179405</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.91337946015</td>
      <td>0.08655102599</td>
      <td>0.00006951385</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.00027410232</td>
      <td>0.98940869350</td>
      <td>0.01031720418</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.99998852246</td>
      <td>0.00001142332</td>
      <td>0.00000005422</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Accuracy:",metrics.accuracy_score(Y_test, DTC_y_pred))
#Clearly has less accuracy compared to logistic regression model
```

    Accuracy: 0.8888888888888888



```python
# Let's evaluate the model based on predictions generated above

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(Y_test,DTC_y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93        14
               1       0.80      0.92      0.86        13
               2       1.00      0.78      0.88         9
    
        accuracy                           0.89        36
       macro avg       0.91      0.88      0.89        36
    weighted avg       0.90      0.89      0.89        36
    



```python
#The precision, f1-score, recall are also less compared to logistic regression model
```


```python
auc = multiclass_roc_auc_score(Y_test,DTC_y_pred, average="macro")
print("Area under curve : ", auc)
#Auc is less when compared to Logistic regression but 0.9 is not a bad value
```

    Area under curve :  0.9089228002271481



```python
# Random Forest is an ensemble technique which uses decision trees as the base learners it has if else loops which are time consuming
```


```python
# Checking Random Forest Classifier
```


```python
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
RFC = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
RFC.fit(X_train,Y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
# Actual class predictions
RF_predictions = RFC.predict(X_test)
# Probabilities for each class
RF_probs = RFC.predict_proba(X_test)[:, 1]
```


```python
#Creating a pandas dataframe for Decision tree result
DTC_results_df = pd.DataFrame(RF_probs)
test_prob_results_df["Predicted"] = RF_predictions 
test_prob_results_df["Actual"] = test_y
test_prob_results_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>Predicted</th>
      <th>Actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00124438217</td>
      <td>0.14541987367</td>
      <td>0.85333574416</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00024930729</td>
      <td>0.99177917166</td>
      <td>0.00797152105</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.90856123347</td>
      <td>0.09142626903</td>
      <td>0.00001249749</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.11310764941</td>
      <td>0.87362844791</td>
      <td>0.01326390267</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.99991003185</td>
      <td>0.00008964675</td>
      <td>0.00000032141</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00030199243</td>
      <td>0.13186040241</td>
      <td>0.86783760516</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.01118626377</td>
      <td>0.97384264313</td>
      <td>0.01497109310</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.99999882476</td>
      <td>0.00000117515</td>
      <td>0.00000000010</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.00012242006</td>
      <td>0.12428350881</td>
      <td>0.87559407113</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.00112897634</td>
      <td>0.87612219227</td>
      <td>0.12274883139</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.99941176716</td>
      <td>0.00058346551</td>
      <td>0.00000476734</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.19064412846</td>
      <td>0.80681374299</td>
      <td>0.00254212855</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.00041863268</td>
      <td>0.99630260756</td>
      <td>0.00327875976</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.99905848721</td>
      <td>0.00094146509</td>
      <td>0.00000004770</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.01729747896</td>
      <td>0.97969396851</td>
      <td>0.00300855253</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.00059131657</td>
      <td>0.97758738230</td>
      <td>0.02182130113</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.00000540635</td>
      <td>0.06687279959</td>
      <td>0.93312179405</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.91337946015</td>
      <td>0.08655102599</td>
      <td>0.00006951385</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.00027410232</td>
      <td>0.98940869350</td>
      <td>0.01031720418</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.99998852246</td>
      <td>0.00001142332</td>
      <td>0.00000005422</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Accuracy:",metrics.accuracy_score(Y_test,RF_predictions))
#Random forest also has accuracy value similar to logistic regression model
```

    Accuracy: 0.9444444444444444



```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test,RF_predictions))
print(classification_report(Y_test,RF_predictions))
print(accuracy_score(Y_test,RF_predictions))

```

    [[13  1  0]
     [ 1 12  0]
     [ 0  0  9]]
                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93        14
               1       0.92      0.92      0.92        13
               2       1.00      1.00      1.00         9
    
        accuracy                           0.94        36
       macro avg       0.95      0.95      0.95        36
    weighted avg       0.94      0.94      0.94        36
    
    0.9444444444444444



```python
auc = multiclass_roc_auc_score(Y_test,RF_predictions, average="macro")
print("Area under curve : ", auc)
```

    Area under curve :  0.9604525908873734



```python
# All the values like accuracy, f1score, recall, precision are same as logistic regression outputs which makes random forest also a good model to classify the Cultivar_Types
```
