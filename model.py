import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import warnings
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import seaborn as sns

# Set random seed for reproducibility
seed = 42

# Load the dataset
df = pd.read_csv(r"D:\Documents\CropPrediction\archive\Crop_recommendation.csv")

warnings.filterwarnings('ignore')


sns.set_style("whitegrid", {'axes.grid' : False})
pio.templates.default = "plotly_white"

# Define target variable and split data
target = 'label'
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Spot-Check Algorithms
models = [
    ('LR', LogisticRegression(solver='saga', max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(probability=True))
]

# Define ensemble models
ensembles = [
    ('AB', AdaBoostClassifier()),
    ('GBM', GradientBoostingClassifier()),
    ('RF', RandomForestClassifier()),
    ('Bagging', BaggingClassifier()),
    ('ET', ExtraTreesClassifier())
]

# Function to perform hyperparameter tuning
def perform_hyperparameter_tuning(model, X_train, y_train):
    return model  # Placeholder, implement grid search here

# Function to perform feature selection
def select_features(model, X_train, y_train):
    return X_train  # Placeholder, implement feature selection here

# Function to train a model
def train_model(model, X_train, y_train):
    pipeline = make_pipeline(StandardScaler(), model)
    trained_model = pipeline.fit(X_train, y_train)
    return trained_model

# Function to evaluate model performance
def evaluate_model(trained_model, X_test, y_test):
    y_pred = trained_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Model: {type(trained_model).__name__}")
    print(f"Training Accuracy Score: {trained_model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {trained_model.score(X_test, y_test) * 100:.1f}%")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print("    ", " ".join(f"({i})" for i in range(conf_matrix.shape[0])))
    for i, row in enumerate(conf_matrix):
        print(f"({i})", end=" ")
        for value in row:
            print(f"{value:4}", end=" ")
        print()
    return trained_model


# Step 1: Combine Models and Train
all_models = models + ensembles
trained_models = {}

for model_name, model_instance in all_models:
    # Step 2: Hyperparameter Tuning
    tuned_model = perform_hyperparameter_tuning(model_instance, X_train, y_train)

    # Step 3: Feature Selection
    selected_features = select_features(tuned_model, X_train, y_train)

    # Step 4: Train Models
    trained_model = train_model(tuned_model, selected_features, y_train)
    trained_models[model_name] = trained_model

# Step 5: Ensemble Methods (Stacking)
stacked_model = StackingClassifier(estimators=all_models, final_estimator=LogisticRegression())

# Step 6: Evaluate Model Performance
for model_name, trained_model in trained_models.items():
    evaluate_model(trained_model, X_test, y_test)

# Step 7: Fit the Stacked Model
stacked_model.fit(X_train, y_train)

# Step 8: Evaluate the Stacked Model
evaluate_model(stacked_model, X_test, y_test)

# Step 9: Select the Best Model
best_model = stacked_model  # Placeholder, select the best model based on performance

# Step 10: Fine-Tuning and Validation (if needed)

# Step 11: Save the Overall Model
def save_model(model,filename):
    pickle.dump(model, open(filename, 'wb'))

# save model
save_model(best_model, 'overall_model.pkl')