# Table of Content
- [Introduction](#introduction)
- [Research Methodology](#research-methodology)
  - [Dataset](#dataset)
  - [Dataset Exploration](#dataset-exploration)
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
  - [Machine Learning Models](#machine-learning-models)
  - [Betting Strategy](#betting-strategy)
- [Results and Evaluation](#results-and-evaluation)
  - [Model Performance](#model-performance)
  - [Profitability Comparison](#profitability-comparison)
- [Conclusion](#conclusion)
- [Further Discussion](#further-discussion)

# Introduction
In sports betting, we know that bookmakers would offer odds that are more favorable to them, i.e. they 
will have a positive expected result. Our intention is to predict or even replicate their attitude towards 
each match and base on it to generate a positive expected model. The result can then be used to construct 
strategies that would make profits potentially.

We assume each match is an independent and mutually exclusive event. Also, we will only use the 
closing odds i.e. last odds before the match starts, as our input due to lack of computing resources for 
building a dynamic predicting model. Through our machine learning model: Gaussian Naïve Bayes 
Classifier, K-Nearest Neighbour Classifier, Multi-layer Perceptron/Feedforward Neural Network, 
Logistic Regression, we will train and test to generate the probabilities of three labels as our output. 
Eventually, we will evaluate the model by their accuracy since the predicted class probabilities will be 
directly used in our betting strategies.


# Research Methodology
## Dataset
The data used in this research comprises the EPL matches spanning 4 seasons, starting from 2019/2020 
till 2022/2023. The data was collected from an online data source. The football-data.co.uk provides the 
historical football matches statistics. After performing data preprocessing, we obtained 1520 matches with 
14 features, which are ‘Referee’, ‘MaxCH’, ‘MaxCD’, ‘MaxCA’, ‘AvgCH’, ‘AvgCD’, ‘AvgCA’, 
‘MaxC>2.5’, ‘AvgC>2.5’, ‘MaxCAHH’, ‘AvgCAHH’, ‘MaxCAHA’, ‘AvgCAHA’, ‘FTR’.
Since we are using closing odds(last odds before match starts), we only select the features with ‘C’, below 
is the description of the features:

![螢幕擷取畫面 (236)](https://github.com/user-attachments/assets/4bb3e8d3-1698-40aa-a1ca-ba010127b34a)

## Dataset Exploration
![image](https://github.com/user-attachments/assets/56b1f94d-4171-4c04-a441-a367b34f2d90)

The number of Home Win(H) and Away Win(A) are similar but the number of Draw(D) is almost half of 
H and A. This is reasonable as the home team has more fans support and is used to the playground, and 
hence has a slightly higher chance, i.e. 43.62% to win a match than the away team, i.e. 33.36% while a 
draw is less likely to happen in football, i.e. 23.03%.

![image](https://github.com/user-attachments/assets/d93862d9-5961-46b6-b807-e987d4c5650e)

In the correlation matrix, we can see that market maximum home win odds(MaxCH) and market 
maximum Asian handicap home team odds(MaxCAHH) are negatively related which is counterintuitive.  
In addition, we observed that the four Asian handicap odds have a weak correlation with market odds.  
We assume bookmakers price them with different information, and hence, they will be included to 
increase performance of the model.

## Data Cleaning and Preprocessing
![image](https://github.com/user-attachments/assets/99fb1e98-d632-4d79-aacc-69357fefaee8)

Since no features contain null values or outliers(a relatively large or small value of odds is reasonable as 
long as the target result is less or more desirable), no data cleaning has to be done. There are 2 categorical 
features needed to be encoded: Referee, FTR. Since the data in the features do not show any ordering nor 
binary, we one-hot encoded the 2 features. After encoding, the number of columns increased to 44. 
 
Before doing the experiment, we need to standardize data by sklearn.preprocessing.StandardScaler, then 
split the data into 80% training data and 20% testing data before feeding into the models. The purpose for 
doing these is to prevent features with wider ranges from dominating the distance metric in different 
models.

## Machine Learning Models
4 classification models: Gaussian Naive Bayes Classifier, K-Nearest Neighbour 
Classifier, Logistic Regression and Multi-layer Perceptron(MLP)/Feedforward Neural Network were trained. All 
models are trained for multiclass classification.

## Betting Strategy
After classification, we would like to generate a profitable strategy given that our models outperform 
random-guessing and historically home team winning percentages. 
 
Denote the predicted class probability of home team win, away team win and draw be 𝑃(𝐻),𝑃(𝐴),𝑃(𝐷) 
respectively, these probabilities would be obtained from various model predictions. Also, denoted the 
average closing odds with corresponding labels: AvgCH, AvgCA, AvgCD by 𝑂<sub>𝐻</sub>, 𝑂<sub>𝐴</sub>, 𝑂<sub>𝐷</sub> respectively. Let 𝐼<sub>𝐶</sub> ={1: 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑 𝑐𝑙𝑎𝑠𝑠=𝑙𝑎𝑏𝑒𝑙, 0: 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒} 𝑓𝑜𝑟 𝐶 ={′𝐴′,′𝐷′,′𝐻′}.
 
Now, we would like to explore if any arbitrage opportunities exist by creating a portfolio with the 
predicted class probabilities and for each game 𝑖, we denoted the gain of above strategy by 𝛼<sub>𝑖</sub> i.e. 𝛼<sub>𝑖</sub> =
𝑂<sub>𝐻</sub>𝑃(𝐻)𝐼<sub>𝐻</sub> +𝑂<sub>𝐴</sub>𝑃(𝐴)𝐼<sub>𝐴</sub> +𝑂<sub>𝐷</sub>𝑃(𝐷)𝐼<sub>𝐷</sub> subjected to 𝑃(𝐻), 𝑃(𝐴), 𝑃(𝐷) > 0  and 𝑃(𝐻) + 𝑃(𝐴)+ 𝑃(𝐷) = 1 . The cost, 𝑐<sub>𝑖</sub> for every game is assumed to be 1 hypothetically since 𝑃(𝐻) + 𝑃(𝐴) + 𝑃(𝐷) = 1. The 
accumulated profit for 𝑛 games is then denoted by 𝜋=∑ <sup>n</sup> <sub>i=1</sub> 𝛼<sub>𝑖</sub> − 𝑐<sub>𝑖</sub>.

# Results and Evaluation

## Model Performance
![螢幕擷取畫面 (237)](https://github.com/user-attachments/assets/6d48091b-f97b-4ee8-ae0c-d49bbee19f6b)

From the table above, we observe that almost all metrics of the Logistic Regression Model are the highest 
among the 4 models. However, its TPR of classifying ‘Draw’ is the lowest among the models, only 
0.26%. Also, we observe that almost all metrics of the Gaussian Naive Bayes Classifier are the lowest but 
its TPR of classifying ‘Draw’ is 0.41%, higher than that of the Logistic Regression Model. 
 
In this stage, we can conclude that the Logistic Regression seems to be the best classifier among the 4 
models as it has the highest metrics values. However, we need to check the profitability of the model by 
the simulation below, and see whether it is still the best model.

## Profitability Comparison
![螢幕擷取畫面 (238)](https://github.com/user-attachments/assets/8f8f157b-2910-49a0-a433-422de492b0fe)

From the table above, we observed that none of our models give a positive average return, even the 
Logistic Regression Model has the lowest average return among the 4 models, which contradicts with the 
above model results. 
 
In football betting, ‘Home Win’ usually has a low odds as Home Team is more advantageous due to the 
familiarity of the stadium and more fans support, which is known as ‘Home Field Advantage’. In contrast, 
‘Away Win’ and ‘Draw’ is rare compared to ‘Home Win’, thus has a higher odds. However, the accuracy 
of predicting ‘Away Win’ and ‘Draw’ is low, which means if we invest into ‘Away Win’ or ‘Draw’, we 
will probably lose a lot of money. If we invest into ‘Home Win’, we can only win a small amount of 
money. Therefore, this is the reason why the 4 models all give a negative average return as we win less in 
‘Home Win’ but lose much more in ‘Away Win’ and ‘Draw’.

# Conclusion
In this project, we have two main goals: predict match results and simulate a profitable strategy. 
 
We use various machine learning models to predict results as doing a multi-class classification project. 
From our research, the soccer game result is predictable by closing odds compared to random guessing. 
We experimented with 4 models: Gaussian Naive Bayes Classifier, K-Nearest Neighbour Classifier, 
Logistic Regression and Multi-layer Perceptron(MLP)/Feedforward Neural Network and all of them give 
a higher accuracy than random guessing. 
 
Another goal is to find a profitable betting strategy using the predicted class probabilities of the above 4 
models. By simulation of 1000 epochs, we observed that none of the models pass our benchmarks and 
they all give a negative average return.   
 
We concluded that our 4 models work well with the classification problem but they are not profitable with 
our proposed strategy. We suggested that closing odds are engineered numbers provided by bookmakers 
so we may know the attitude of bookmakers toward the game but we cannot arbitrage on them with the 
odds they offered.

# Further Discussion
For further improvement on classification, we may not choose Logistic Regression, instead we may 
choose Gaussian Naive Bayes Classifier as it has a higher TPR of classifying a ‘Draw’ and balance TPR 
of classifying ‘Home Win’ and ‘Away Win’. Therefore, it can have a more favorable return on both 
‘Home Win’ and ‘Away Win’. Moreover, more features can be added so as to increase the accuracy of the 
models, such as the condition of the stadium and the players, weather, number of fans for home team and 
away team respectively. 
 
For further improvement on simulation, we suggested that we may bet with a learnable bet size. In this 
project, we assume a fixed amount of money to bet on in every game. Instead, this could also turn into a 
learnable parameter in the simulation by setting some thresholds on features of datasets. By executing 
such a strategy, we hope to enlarge the profits of betting on a strong team that has a higher chance to win
