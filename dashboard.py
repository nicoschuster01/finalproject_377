import streamlit as st
import pandas as pd
from PIL import Image

# Load data
data = pd.read_csv("randooutput/fraudie_sampled.csv")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Report"])

    if page == "Home":
        st.title('Predicting Fraud in Financial Transactions')
        img = Image.open('pics/photo.jpeg')
        st.image(img)

        st.header('Data Overview')
        """
        For this project, we want to create a predictive model to detect **fraud in financial service payments and transfers**. Currently, there are many hurdles when it comes to fraud detection; many of the predictors are fallible and don't correctly predict whether a payment is actually fraudelent or not. The goal of this project will be to train several different machine learning models to detect fraud given the attributes: "_payment_type_", "_amount_", "_nameOrig_", "_oldBalanceOrig_", "_namedDest_", "_isFlaggedFraud_", "_isFraud_", among others.
        """
        
        st.write(data.head(10))
        
        st.header('Correlation Matrix')
        """
        We took the data prior to any preprocessing and ran them through several visual explorations. We can see from the correlation matrix presented that there weren't any overwhelming relationships between any of the existing numerical features and a transaction being labeled as fraudelent. This indicates that either the existing features in the dataset have virtually no predictive power of our outcome variable (*isFraud*) or that our predictor variables exhibit multicollinearity which would make it difficult to attribute a strong predictive effect to any one variable.
        """
        img = Image.open('pics/plot1.png')
        st.image(img)

        st.header('Data Preprocessing/Parameter Optimization')
        st.code("""
        # Set random seed for reproducibility
rng = np.random.RandomState(0)
sample_size = 100000
# Proportion of fraudulent transactions in the sample
fraud_proportion = 0.2  # Adjust as needed

# Sample a subset of the data including both nonfraud and fraud transactions
fraudie_fraud = fraudie[fraudie['isFraud'] == 1]
fraudie_non_fraud = fraudie[fraudie['isFraud'] == 0].sample(int(sample_size * (1 - fraud_proportion)), random_state=rng)
fraudie_sampled = pd.concat([fraudie_fraud, fraudie_non_fraud], axis=0)

# Shuffle dataset
fraudie_sampled = fraudie_sampled.sample(frac=1, random_state=rng)

os.makedirs('randooutput', exist_ok=True)
fraudie_sampled.to_csv('randooutput/fraudie_sampled.csv', index=False)

# Split off X and y from the sampled dataset
y_sampled = fraudie_sampled['isFraud']
X_sampled = fraudie_sampled.drop('isFraud', axis=1)

# Split the sampled data into training and testing sets
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=rng)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['type', 'nameOrig', 'nameDest']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X_sampled.select_dtypes(include=['int64', 'float64']).columns),
    ('cat', categorical_transformer, categorical_features)
])

# Model selection pipeline with Random Forest
model = RandomForestClassifier(random_state=0)
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', model)
])

# Hyperparametrs
param_dist = {
    'model__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'model__max_depth': [None, 10, 20],  # Maximum depth of the trees
    'model__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'model__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
}

random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train_sampled, y_train_sampled)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
""", language='python')
        st.subheader('Results')
        """
        **Best parameters**: {'model__n_estimators': 300, 'model__min_samples_split': 10, 'model__min_samples_leaf': 1, 'model__max_depth': None}

**Best score**: 0.9832081645113453
"""
        st.subheader('Predicting on Target Variable')
        st.code("""
        # Predict on test set
y_pred = random_search.predict(X_test_sampled)

# Compute evaluation metrics
accuracy = accuracy_score(y_test_sampled, y_pred)
precision = precision_score(y_test_sampled, y_pred)
recall = recall_score(y_test_sampled, y_pred)
f1 = f1_score(y_test_sampled, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")""")
        """
        **Accuracy**: 0.9850
        
        **Precision**: 0.9910
        
        **Recall**: 0.8501
        
        **F1-score**: 0.9151
"""
        st.header('Confusion Matrix')
        img = Image.open('pics/plot2.png')
        st.image(img)
        """
        These overwhelming statistics indicated that there was still a large class imbalance present when it came to whether a transaction was indeed fraud or not. The limitations shined through and yielded a model that was incredibly good at predicting whether a payment was fraudelent or not by simply labeling most transactions as such (after all, most of the transactions in our dataset were non-fraudelent to begin with). However, it still managed to yield a suprisingly good precision which is important in the realm of fraud detection because the number of false positives is typically pretty high. 
        """

        st.header('Precision Recall Curve')
        #st.subheader('My sub')
        img = Image.open('pics/plot3.png')
        st.image(img)

        st.header('ROC Curve')
        #st.subheader('My sub')
        img = Image.open('pics/plot4.png')
        st.image(img)
        
        st.code('')

    if page == "Report":
        st.title("Report: Predicting Fraud in Financial Transactions")
        """
### By: Avi Gadde and Nico Schuster

## Research

### Research Question

For this project, we want to create a predictive model to detect **fraud in financial service payments and transfers**. Currently, there are many hurdles when it comes to fraud detection; many of the predictors are fallible and don't correctly predict whether a payment is actually fraudelent or not. The goal of this project will be to train several different machine learning models to detect fraud given the attributes: "_payment_type_", "_amount_", "_nameOrig_", "_oldBalanceOrig_", "_namedDest_", "_isFlaggedFraud_", "_isFraud_", among others. 

### Hypothesis

The application of **machine-learning algorithms** will effectively predict fraudulent transactions in financial payment services with **high accuracy**.

### Measurement of Success

Our project is centered around predicting fraud in financial transactions using a machine learning algorithm. In this realm, success is measured by the machine-learning model’s ability to accurately predict fraudulent transactions based on given test data after it has been trained using training data.

In addition to accuracy, success will be measured with **precision**, which is the proportion of true positive identifications made overall positive identifications. Furthermore, it will be measured with **recall** (sensitivity) which is the proportion of actual positives correctly identified. Lastly, we can also measure success with the **F1 score**, being a mean of precision and recall and finding a balance between the two.

### Dashboard

The dashboard will feature various visualizations to provide insights into financial transaction data and fraud occurrences. Bar charts will display transaction type distribution, while pie charts will visualize the proportion of fraudulent versus non-fraudulent transactions. Line charts will track fraud occurrences over time, and scatter plots will explore relationships between features like transaction amount and fraud occurrence. These visualizations will aid in understanding the dataset and detecting future fraudulent transactions.

## Necessary Data

### Data Collection

The data that we are utilizing for the project is a *synthetic dataset* and was created with the sole purpose of fraud detection analysis. The data was created to simulate real-world financial transaction data and includes everything we need to formulate a machine-learning algorithm in order to develop a predictive model. This ranges from detailed transaction data, customer profiles, fraudulent patterns, transaction amounts, and merchant information.

The final dataset will be compressed from the original dataset that contains over **six million rows** with a high-class imbalance. By cleaning and balancing the dataset accordingly through feature engineering, we hope to create a model that will accurately (in a relative sense) predict whether or not a designated transfer is indeed fraudulent or not.

### About the Dataset

The main **observation unit** in this dataset would be *nameDest*, with a unique value count of roughly 2.7 million. We will be treating this attribute as the entity that is being checked for fraudulent activity. 

#### Inputs

Our **raw inputs** are step count, type of transaction, amount of transaction, ID of the customer who initiated the transaction, initial balance before the transaction, customer’s balance after the transaction, the recipient ID, initial recipient balance before the transaction, recipient’s balance after the transaction, identification of if the transaction could potentially be a fraud transaction or not, and flags of illegal attempts to transfer more than 200,000 in a single transaction. The inputs are obtained from the dataset which is stored in a file named “**financial_transaction_data**.”

The input files will be sorted in a folder called “_Inputs_” so they will be easily accessible. Additionally, if along the way, we discover more valuable information/datasets that we can take advantage of, we can download them and store them in the inputs folder making it a smooth and organized process.

The variables that are **absolutely necessary** are the amounts of the transactions, the IDs of both of the subjects involved in the transactions, the new and old account balances of the subjects, and if fraud was potentially identified.

As for the variables that we would like to have if possible are most of the ones given in the dataset. That being: _type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, and isFlaggedFraud._

#### Sample

Since there is a scarcity of large public datasets containing financial service payments, we are working with a synthetic dataset created using a simulator. Thus, this dataset is derived from a sample of transactions from financial logs over a **one-month period**. These financial logs were provided by a multinational company that has a presence in 14 countries worldwide. 

As it is without any cleaning, we are dealing with **668,670 rows** with **11 columns** as previously mentioned. Most notably, there is currently a staggering class imbalance in the ‘_isFraud_’ column of the dataset. When taking the value counts we see that there are **404 rows** that are actually fraudulent while the other **668,266** are nonfraudulent, roughly equivalent to a **0.06%** quantity of fraudulent transactions in the entire dataset. In the same breadth, we see that the ‘isFlaggedFraud’ column has a value count of **668,670** equal to zero. So none of these transactions are appropriately predicted either. In order to remedy this, we’ll have to select and transform the most relevant attributes in the dataset to help predict fraudulent transactions.

### Speculation

To transform the raw data into something usable we’ll have to employ a variety of different tactics to clean up the class imbalance and make actual use of the features as predictors for whether or not a payment/transfer is fraudulent. To accomplish this, we will preprocess the data and scale it accordingly so that it’s ready for use when we finally decide to employ different decision tree approaches. One way to do this is through the use of the **SMOTE class** (Synthetic Minority Oversampling Technique/k-Nearest Neighbors). Since we think most of the issues are arising from the poor performance of the minority class, using this technique will help mitigate the class imbalance and thus yield more accurate results despite the limitations of the dataset.
        """
        
main()