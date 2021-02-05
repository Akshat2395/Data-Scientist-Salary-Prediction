# Data-Scientist-Salary-Prediction

## Introduction:

Kaggle hosted an open data scientist competition in 2019 titled “2019 Kaggle Machine Learning & Data Science Survey" challenge. The purpose of this challenge was to “tell a data story about a subset of the data science community represented in this survey, through a combination of both narrative text and data exploration.” They set out to conduct an industry-wide survey that presents a truly comprehensive view of the state of data science and machine learning. More information on the competition, data, and prizes can be found on: https://www.kaggle.com/c/kaggle-survey-2019

Using this survey data, this project sets out to build a classification model where we use machine learning to classify salaries of individuals based on their current location, experience, knowledge, etc. 

The purspose of this project is to:

1. understand and explore employment in the data science community, as represented in a survey conducted by Kaggle.
2. train, validate, and tune multi-class ordinary classification problem that can classify, given a set of survey responses by a data scientist, what a survey respondent’s current yearly compensation bucket is.

This project outlines:
1. Data Cleaning
2. Exploratory data analysis
3. Feature selection
4. Model implementation
5. Model tuning
6. Testing and discussion

## Project Summary:

### 1. Data Preparation and Cleaning
The original data (multiple_choice_responses.csv) has been transformed to Kaggle_Salary.csv as per the code given in KaggleSalary_DataSet_cleaning.ipynb. In the dataset to be used (Kaggle_Salary.csv- File to be read in notebook for this Project), rows with the null values of salaries have been dropped. The salaries of all respondents are given in the column named 'Q10'. In addition, two columns (‘Q10_Encoded’ and ‘Q10_buckets’) has been added at the end. Column ‘Q10_buckets’(Target Variable) has been obtained by combining some salary buckets in the column ‘Q10’. Column ‘Q10_Encoded’ has been obtained by label encoding the column ‘Q10_buckets'.

**Dropping columns**

By going through all the survey questions, few of them are highly subjective and irrelevant with respect to our target variable and hence can be dropped.

**Imputing missing values**

The mean of the numerical column data is used to replace null values when the data is normally distributed. Median is used if the data comprised of outliers. Mode is used when the data having more occurences of a particular value or more frequent value.

### 2. Data Exploration - Key Findings
**Distribution of gender**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Gender_of_Respondents.png" alt="Gender of Respondents" width="700" height="600">

This plot represents the gender of repondents. From this plot, we can say that the majority were Male with 83.8% of total respondents.


**Gender distribution w.r.t job profile**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Gender_distribution_wrt_job_profile.png" alt="Gender of Respondents" width="1000" height="600">

We can analyze from this plot that the majority of respondents were Data Scientists and males dominated each job profile. In the chart females and males show a similar distribution.


**Level of education**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Education_of_respondents.png" alt="Gender of Respondents" width="1500" height="400">

The graph obtained above represents the number of respondents from various educational backgrounds. We can see that maximum number of respondents have a Master's degree followed by that of Bachelor's degree and Doctoral degree.


**Country Distribution of Data Scientist by Gender**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Country_distrbution_of_data_scientists.png" alt="Gender of Respondents" width="1000" height="600">

The bar plot describes the country distribution of data scientists by gender. The maximum number of data scientists reside in United States of America and the majority of them are males. After USA, India holds the 2nd highest number of data scientists and this role is male dominated there too. We can also see that there is a vast difference in number of respondents who are data scientists in the entire world.


**Salary Distribution of Data Scientist by Gender**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Salary_distribution_among_data_scientists.png" alt="Gender of Respondents" width="1000" height="600">

The above interactive graph represents the yearly compensation of data scientists by their gender. The maximum number of repondents have a yearly compensation of 0-9,999 USD $. We can say that the respondents were not comfortable in sharing their true salary. As the choices for this question were in USD, the conversion of one currency to another might also lead to misclassification of data.


**Age group and sex ratio of the responders**

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Age_vs_Sex_ratio.png" alt="Gender of Respondents" width="1000" height="600">

25-29 and 30-34 age groups gave the most responses in the survey and most of them were males


### 3. Feature Selection

**How does feature engineering useful in machine learning?**

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If feature engineering is done correctly, it increases the predictive power of machine learning algorithms by creating features from raw data that help facilitate the machine learning process.

The problem is that with label encoding, the categories now have natural ordered relationships. The computer does this because it’s programmed to treat higher numbers as higher numbers; it will naturally give the higher numbers higher weights.

1. Correlation Matrix - 

A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses. It is used to summarize a large amount of data where the goal is to see patterns. For our example, the observable pattern will show us how the variables will correlate with each other.
A threhold value of 0.1 and greater gives us 43 columns which are related to the target column.

2. Lasso CV - 

LASSO is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces.
For the same threhold value as 0.1, this method gives us 105 columns

3. PCA - 

As the number of important features selected by LASSO is quite large (105 features), I will be using a dimensionality reduction technique called PCA (Principal component analysis). It is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. Such dimensionality reduction can be a very useful step for visualising and processing high-dimensional datasets, while still retaining as much of the variance in the dataset as possible.

We can find out that by using only 92 columns, we can achieve a cumulative explained variance of 95% which I considered to be enough to build our ML model.

### 4. Model Implementation

Here we will apply Logistic regression to build our classification model. 

**Logistic Regrression**

The logistic model is used to model the probability of a certain class or event existing such as pass/fail or win/lose. Logistic Regression is used when the dependent variable(target) is categorical. There are times where we need to classify whether a certain scenario is considered as class 0 or class 1 which is in the case of a binary classification regression. For such models, linear regression can be used. But for the linear regression to work, there is a need for setting up a threshold based on which classification can be done. For our example, it can be inferred that linear regression is not suitable for classification problem.

Logistic regression is named for the function used at the core of the method, the logistic function. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.

Due to the properties of logistic regression, we cannot directly apply it on our model as it contains more than just 2 classes (15 in our case). The logistic function is only able to classy the data between 2 classes, either 0 or 1. So, we will divide the labels in such a way that we can use logistic regression multiple times, each time classifying between one class vs. the rest. As our target data is ordered, we cannot use one vs. rest method to solve this as it will consider one label as class 0 and the rest of them as class 1.

Simply applying logistic regression for classification results to an accuracy of 38.21%.

### 5. Model Tuning

Hyperparameters are important because they directly control the behaviour of the training algorithm and have a significant impact on the performance of the model being trained. The process of finding the best parameters for our model is called hyperparameter tuning. It can be done by various algorithms but here, we will be using Grid Search which trains the algorithm for all combinations by using the two set of hyperparameters (learning rate and number of layers) and measures the performance using “Cross Validation” technique. This validation technique gives assurance that our trained model got most of the patterns from the dataset. One of the best methods to do validation by using “K-Fold Cross Validation” which helps to provide ample data for training the model and ample data for validations.

Accuracy before tuning:  38.21 %
Accuracy after tuning:  38.4 %

Hyperparameter tuning results in 0.2% increase in accuracy. The accuracy seems low but it is actually acceptable because we are predicting 1 class among 8.

When it comes to classification, accuracy is not considered an ideal metric to identify the performance of the model. 

### 6. Discussion

**Confusion Matrix**

A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.

The confusion matrix is usually useful for binary classification models where the model has to classify between class 0 and class 1. It is extremely useful for measuring Recall, Precision, Specificity, Accuracy and most importantly AUC-ROC Curve. As our model consists of 15 different classes, we can use this technique to determine only the precision, recall and f1-score.

In our dataset, the maximum number of respondents' yearly compensation lie in the range 0-9,999 USD which outnumbers all the other salary ranges. Due to this, the target data becomes skewed as there are more number of class 0's and quite less for other classes. As seen from the confusion matrix, the relation between true value of label 0 and predicted value of label 0 is very high in comparison to other predictions. We can also see that the recall scores for all the classes except for class 0 are quite low. Hence the model suffers from high bias.


### 7. Conclusion 

The accuracy can be increased by further grouping the salary brackets. Also, other feature selection techniques such as chi square can be applied to check the affect on accuracy and tuning the respective hyper-parameters.

The accuracy can be further increased if the analysis would have been done per country, this would reduce the variance in the data and errors being added due to different money buying power in different countries. This creates an error of perspective, as the buying power of 1 dollar is differnt in Canada when compared to India
