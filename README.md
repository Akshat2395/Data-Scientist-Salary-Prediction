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

<img src="https://github.com/Akshat2395/Data-Scientist-Salary-Prediction/blob/main/images/Gender_of_Respondents.png" alt="Gender of Respondents" width="400" height="400">

This plot represents the gender of repondents. From this plot, we can say that the majority were Male with 83.8% of total respondents.

****
