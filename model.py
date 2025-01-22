
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
import joblib

# Load data
input_file = '/Users/morganakinlan/Desktop/nlp_smote/hierarchy.xlsx'
df = pd.read_excel(input_file)

# Features and targets
X = df[['DEPT_NUM','COMP_OFFICE_SERVICE', 'REGION']]
y = df[['AUDIT_CAT', 'TEAM']]

# Ensure all relevant columns are treated as strings
X = X.astype(str)
y = y.astype(str)

# Handle missing values
X = X.fillna('')
y = y.fillna('')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#pipeline using imblearn, not sklearn Pipeline
pipeline_audit_cat = imbpipeline([
    ('vectorizer', TfidfVectorizer()),
    ('sm',  SMOTE()),
    ('classifier', RandomForestClassifier())
])

#Use ColumnTransformer to apply TfidfVectorizer to each text column separately for TEAM
preprocessor_team = ColumnTransformer(
    transformers=[
        ('comp', TfidfVectorizer(), 'COMP_OFFICE_SERVICE'),
        ('reg', TfidfVectorizer(), 'REGION')
    ]
)

pipeline_team = imbpipeline([
    ('preprocessor', preprocessor_team),
    ('sm',  SMOTE()),
    ('classifier', RandomForestClassifier())
])


class Basement(BaseEstimator): #Because the basement is where the pipes are :) 
    def __init__(self,pipeline_1,pipeline_2):
        self.pipeline_1 = pipeline_1
        self.pipeline_2 = pipeline_2

    def fit_pipes(self,X_train_col1,y_train_col1,X_train_cols,y_train_col):
        # Fit both pipelines
        self.pipeline_1.fit(X_train_col1, y_train_col1)  
        self.pipeline_2.fit(X_train_cols, y_train_col)
        
    def predict_pipes(self,X_test):
        y_pred_audit_cat = self.pipeline_1.predict(X_test['COMP_OFFICE_SERVICE'])  # predicting AUDIT_CAT column
        y_pred_team = self.pipeline_2.predict(X_test[['COMP_OFFICE_SERVICE', 'REGION']]) #predicting TEAM column
        preds = pd.DataFrame({
            'AUDIT_CAT': y_pred_audit_cat,
            'TEAM': y_pred_team
        })
        return preds
    
    def rejoin(self,df,preds):
        results = pd.merge(df,preds,left_index=True, right_index=True)
        return results
        

audit_cat_X_train= X_train['COMP_OFFICE_SERVICE']
audit_cat_y_train = y_train['AUDIT_CAT']
team_X_train = X_train[['COMP_OFFICE_SERVICE', 'REGION']]
team_y_train = y_train['TEAM']

pipes_model = Basement(pipeline_audit_cat,pipeline_team) #Instantiate the class, these will be pipeline_1 and pipline_2 in the class we created above
pipes_model.fit_pipes(audit_cat_X_train,audit_cat_y_train, team_X_train,team_y_train)
#predictions = pipes_model.predict_pipes(X_test)
#print(predictions)


joblib.dump(pipes_model, 'hierarchy_prediction_model.pkl') #serialize model weights for future use

print("Model saved as hierarchy_prediction_model.pkl")

