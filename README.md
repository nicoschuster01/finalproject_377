# Research Proposal: Predicting Fraud in Financial Transactions
### By: Avi Gadde and Nico Schuster

## Research

### Research Question

For this project, we want to create a predictive model to detect fraud in financial service payments and transfers. Currently, there are many hurdles when it comes to fraud detection; many of the predictors are fallible and don't correctly predict whether a payment is actually fraudelent or not. The goal of this project will be to train several different machine learning models to detect fraud given the attributes: *'payment_type'*, *'amount'*, *'nameOrig'*, *'oldBalanceOrig'*, *'namedDest'*, *'isFlaggedFraud'*, *'isFraud'*, among others. 

### Hypothesis

The application of machine learning algorithms will effectively predict fraudulent transactions in financial payment services with high accuracy.

### Measurement of Success

Our project is centered around predicting fraud in financial transactions using a machine learning algorithm. In this realm, success is measured by the machine-learning model’s ability to accurately predict fraudulent transactions based on given test data after it has been trained using training data.

In addition to accuracy, success will be measured with precision, which is the proportion of true positive identifications made overall positive identifications. Furthermore, it will be measured with recall (sensitivity) which is the proportion of actual positives correctly identified. Lastly, we can also measure success with the F1 score, being a mean of precision and recall and finding a balance between the two.

## Necessary Data

### Data Collection

The data that we are utilizing for the project is a synthetic dataset and was created with the sole purpose of fraud detection analysis. The data was created to simulate real-world financial transaction data and includes everything we need to formulate a machine-learning algorithm in order to develop a predictive model. This ranges from detailed transaction data, customer profiles, fraudulent patterns, transaction amounts, and merchant information.

The final dataset will be compressed from the original dataset that contains over six million rows with a high-class imbalance. By cleaning and balancing the dataset accordingly through feature engineering, we hope to create a model that will accurately (in a relative sense) predict whether or not a designated transfer is indeed fraudulent or not.

### About the Dataset

The main observation unit in this dataset would be *'nameDest'*, with a unique value count of roughly 2.7 million. We will be treating this attribute as the entity that is being checked for fraudulent activity. 

#### Inputs

Our raw inputs are step count, type of transaction, amount of transaction, ID of the customer who initiated the transaction, initial balance before the transaction, customer’s balance after the transaction, the recipient ID, initial recipient balance before the transaction, recipient’s balance after the transaction, identification of if the transaction could potentially be a fraud transaction or not, and flags of illegal attempts to transfer more than 200.000 in a single transaction. The inputs are obtained from the dataset which is stored in a file named *"financial_transaction_data"*

The input files will be sorted in a folder called “Inputs” so they will be easily accessible. Additionally, if along the way, we discover more valuable information/datasets that we can take advantage of, we can download them and store them in the inputs folder making it a smooth and organized process.

The variables that are absolutely necessary are the amounts of the transactions, the IDs of both of the subjects involved in the transactions, the new and old account balances of the subjects, and if fraud was potentially identified.

As for the variables that we would like to have if possible are most of the ones given in the dataset. That being: *'payment_type'*, *'amount'*, *'nameOrig'*, *'oldBalanceOrig'*, *newbalanceOrig*, *'namedDest'*, *oldbalanceDest*, *newbalanceDest*, *'isFlaggedFraud'*, *'isFraud'*.

#### Sample

Since there is a scarcity of large public datasets containing financial service payments, we are working with a synthetic dataset created using a simulator. Thus, this dataset is derived from a sample of transactions from financial logs over a one month period. These financial logs were provided by a multinational company that has a presence in 14 countries worldwide. 

As it is without any cleaning, we are dealing with 668,670 rows with 11 columns as previously mentioned. Most notably, there is currently a staggering class imbalance in the *‘isFraud’* column of the dataset. When taking the value counts we see that there are 404 rows that are actually fraudulent while the other 668266 are nonfraudulent, roughly equivalent to a 0.06% quantity of fraudulent transactions in the entire dataset. In the same breadth, we see that the *‘isFlaggedFraud’* column has a value count of 668670 equal to zero. So none of these transactions are appropriately predicted either. In order to remedy this, we’ll have to select and transform the most relevant attributes in the dataset to help predict fraudulent transactions.

### Speculation

To transform the raw data into something usable we’ll have to employ a variety of different tactics to clean up the class imbalance and make actual use of the features as predictors for whether or not a payment/transfer is fraudulent. To accomplish this, we will preprocess the data and scale it accordingly so that it’s ready for use when we finally decide to employ different decision tree approaches. One way to do this is through the use of the SMOTE class (Synthetic Minority Oversampling Technique/k-Nearest Neighbors). Since we think most of the issues are arising from the poor performance of the minority class, using this technique will help mitigate the class imbalance and thus yield more accurate results despite the limitations of the dataset.
