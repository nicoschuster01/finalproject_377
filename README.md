# Research Proposal: Predicting Fraud in Financial Transactions
### By Avi Gadde and Nico Schuster

## Research

#### Research Question

For this project, we want to create a predictive model to detect fraud in financial service payments and transfers. Currently, there are many hurdles when it comes to fraud detection; many of the predictors are fallible and don't correctly predict whether a payment is actually fraudelent or not. The goal of this project will be to train several different machine learning models to detect fraud given the attributes: payment_type, amount, nameOrig, oldBalanceOrig, namedDest, isFlaggedFraud, isFraud, among others. 

#### Hypothesis

The application of machine learning algorithms will effectively predict fraudulent transactions in financial payment services with high accuracy.

#### Measurement of Success

Our project is centered around predicting fraud in financial transactions using a machine learning algorithm. In this realm, success is measured by the machine learning model’s ability to accurately predict fraudulent transactions based on given test data after it has been training using training data.

In addition to accuracy, success will be measured with precision, which uss the proportion of tru positive identifications made over all positive identifications. Furthermore, it will be measured with recall (sensitivity) which is the proportion of actual positives correctly identified. Lastly, we can also measure success with the F1 score, being a mean of precision and recall and finding balance between the two.

## Necessary Data¶

#### Data Collection

What data do we have and what data do we need?

How will we collect more data?

What does the final dataset need to look like (mostly dictated by the question and the availability of data):

#### About the Dataset

What is an observation, e.g. a firm, or a firm-year, etc.

What is the sample period?

What are the sample conditions? (Years, restrictions you anticipate (e.g. exclude or require some industries)

What variables are absolutely necessary and what would you like to have if possible?

What are the raw inputs and how will you store them (the folder structure(s) for each input type)?

#### Speculation

Speculate at a high level (not specific code!) about how you’ll transform the raw data into the final form.
