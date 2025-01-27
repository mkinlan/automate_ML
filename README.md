
# Automating Spreadsheet Mapping with Machine Learning

Imagine you have a spreadsheet of department information that you download from Corporate every quarter. Your job is to map these departments to specific audit categories and teams. You've been doing this so long that it's second nature. It doesn't take much time; typically there are only a handful of new departments every quarter. Easy peasy!

Unfortunately, your manager now wants this mapping to be completed weekly, instead of quarterly. Additionally, your company is making several major acquisitions, which will mean many more new departments. This will take considerably more time to do by hand, and will be tedious to boot. 

Is there a way to automate this? 

You know that the departments align to regions and teams in a logical way, but you've never actually needed to write out the rules behind this mapping. Could you use machine learning to learn your unwritten mapping rules?

Let's see! 

### Getting Started

We'll start by importing a number of Python modules and loading the Excel file of our department information. 

To keep your project workspace nice and clean, I recommend creating a virtual environment and then importing the requirements.txt file from this repo.

The main libraries we'll use are: 

* pandas: https://pandas.pydata.org
* scikit-learn: https://scikit-learn.org/1.6/install.html
* imbalanced learn: https://imbalanced-learn.org/stable/install.html  (this relies on scikit-learn, so install that first)
* joblib: https://joblib.readthedocs.io/en/stable/installing.html


```python
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
input_file = '/Users/mkinlan/Desktop/hierarchy.xlsx'
df = pd.read_excel(input_file)

#Print the shape of the dataframe
print("Shape of the dataframe:", df.shape)

df.head()
```

    Shape of the dataframe: (110, 7)





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
      <th>DEPT_NUM</th>
      <th>DEPT_NAME</th>
      <th>COMP_OFFICE_SERVICE</th>
      <th>REG_PROGRAM</th>
      <th>REGION</th>
      <th>AUDIT_CAT</th>
      <th>TEAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Human_Resources_1</td>
      <td>ABC_Corp.Headquarters.Employee_Onboarding</td>
      <td>North_America.Onboarding_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Finance_1</td>
      <td>ABC_Corp.Headquarters.Accounts_Payable</td>
      <td>Europe.Financial_Management</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Information_Technology_1</td>
      <td>ABC_Corp.Tech_Hub.Technical_Support</td>
      <td>Asia.Tech_Support_Program</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Marketing_1</td>
      <td>ABC_Corp.Marketing_Office.Advertising</td>
      <td>South_America.Ad_Campaigns</td>
      <td>REGION_2</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Sales_1</td>
      <td>ABC_Corp.Sales_Department.Customer_Relations</td>
      <td>Africa.Customer_Engagement</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
    </tr>
  </tbody>
</table>
</div>



The target columns that you want to predict will be the AUDIT_CAT (aka audit category) and TEAM columns. 

Although you've never written down the rules for creating these columns, you know which other columns affect them. You remember that the values for AUDIT_CAT depend primarily on the values for COMP_OFFICE_SERVICE (aka company, office, service) and that the values for TEAM depend on the data in the REGION and COMP_OFFICE_SERVICE columns.

This means we can disregard most of the other columns for the predictive model. We'll start by separating our desired columns into targets and features, then doing some light cleaning:

Let's keep the DEPT_NUM column in the list of features, even though it's not really a predictor. That will help us keep track of which department is which.


```python
# Features and targets
X = df[['DEPT_NUM','COMP_OFFICE_SERVICE', 'REGION']]
y = df[['AUDIT_CAT', 'TEAM']]

# Ensure all relevant columns are treated as strings
X = X.astype(str)
y = y.astype(str)

# Handle missing values
X = X.fillna('')
y = y.fillna('')
```

### Pipelines

Next, we'll create several pipelines to process the data for the model. We'll make one pipeline for the AUDIT_CAT column:


```python
#pipeline using imblearn, not sklearn Pipeline
pipeline_audit_cat = imbpipeline([
    ('vectorizer', TfidfVectorizer()),
    ('sm',  SMOTE()),
    ('classifier', RandomForestClassifier())
])
```

The first step in the pipeline is to convert the text data from the column into numeric data using TfidfVectorizer. After the data has been converted to numeric values, we use the SMOTE algorithm to adjust for class imbalance. The AUDIT_CAT column can have one of 4 possible values (CAT_A, CAT_B, CAT_C, or CAT_D). There may not be enough values in each of these 4 classes for the model to learn well, so we'll mitigate this by using SMOTE to synthesize data from the minority class(es).

The final pipeline step is a RandomForestClassifier function. This will classify each observation (aka department) into one of the 4 classes using the Random Forest algorithm.

We'll make a separate pipeline for the TEAM column. Since this target column will be predicted by two features instead of one, it's a little more complex.  We'll break out the TfidifVectorizer step into a separate preprocessing pipeline to make sure it vectorizes each feature correctly. Then we'll take the results of that pipeline and pass them into the main pipeline for the target column:


```python

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

```

Now that we've created the pipelines, we need to split up our data into training and test sets so we can run the training data through our new pipelines.

We'll do a train/test/split to split up our data into 80% training, 20% test data, and take a look at it:


```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Print the shapes of resampled X_train and y_train
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
```

    Shape of X_train: (88, 3)
    Shape of y_train: (88, 2)



```python
#check out the training data just to be sure it's right
print(y_train.head())
print(X_train.head())
```

       AUDIT_CAT        TEAM
    65     CAT_A    RED_TEAM
    26     CAT_A   BLUE_TEAM
    22     CAT_C  GREEN_TEAM
    31     CAT_B    RED_TEAM
    47     CAT_B    RED_TEAM
       DEPT_NUM                          COMP_OFFICE_SERVICE    REGION
    65       66      ABC_Corp.R&D_Center.Product_Development  REGION_3
    26       27  ABC_Corp.Call_Center.Call_Center_Operations  REGION_1
    22       23          ABC_Corp.Tech_Hub.Technical_Support  REGION_3
    31       32       ABC_Corp.Headquarters.Accounts_Payable  REGION_1
    47       48               ABC_Corp.Legal_Dept.Compliance  REGION_1


### Training the Model(s)

Now it's time to run the training data through both pipelines and get our predictions!


```python
# Train both pipelines
pipeline_audit_cat.fit(X_train['COMP_OFFICE_SERVICE'], y_train['AUDIT_CAT'])  
pipeline_team.fit(X_train[['COMP_OFFICE_SERVICE', 'REGION']], y_train['TEAM'])

```




    Pipeline(steps=[('preprocessor',
                     ColumnTransformer(transformers=[('comp', TfidfVectorizer(),
                                                      'COMP_OFFICE_SERVICE'),
                                                     ('reg', TfidfVectorizer(),
                                                      'REGION')])),
                    ('sm', SMOTE()), ('classifier', RandomForestClassifier())])




```python
#Get predictions for AUDIT_CAT column
y_pred_audit_cat = pipeline_audit_cat.predict(X_test['COMP_OFFICE_SERVICE'])  # predicting AUDIT_CAT column
```


```python
#Get predictions for TEAM column
y_pred_team = pipeline_team.predict(X_test[['COMP_OFFICE_SERVICE', 'REGION']]) #predicting TEAM column

```


```python
#Check the predictions. How were they? 
audit_cat_report = classification_report(y_test['AUDIT_CAT'], y_pred_audit_cat, labels=y['AUDIT_CAT'].unique(),
                                           target_names=y['AUDIT_CAT'].unique())
print("Classification report for test data set for AUDIT_CAT:", audit_cat_report)
```

    Classification report for test data set for AUDIT_CAT:               precision    recall  f1-score   support
    
           CAT_B       1.00      1.00      1.00        11
           CAT_C       1.00      1.00      1.00         7
           CAT_A       1.00      1.00      1.00         4
    
        accuracy                           1.00        22
       macro avg       1.00      1.00      1.00        22
    weighted avg       1.00      1.00      1.00        22
    



```python
team_report = classification_report(y_test['TEAM'], y_pred_team, labels=y['TEAM'].unique(),
                                    target_names=y['TEAM'].unique())
print("Classification report for test data set for TEAM:", team_report)
```

    Classification report for test data set for TEAM:               precision    recall  f1-score   support
    
        RED_TEAM       1.00      1.00      1.00        16
      GREEN_TEAM       1.00      1.00      1.00         3
       BLUE_TEAM       1.00      1.00      1.00         3
    
        accuracy                           1.00        22
       macro avg       1.00      1.00      1.00        22
    weighted avg       1.00      1.00      1.00        22
    


### Combine and Pickle It!

This was great, but it's not very usable. 

We've created two separate models, one for each column we want to predict. What if we combined both pipelines into one? Then we would only need to call a single model when we wanted to reuse it on new data.

Let's create a class with a function to fit both pipeslines in one go. Then, when we create a new instance of this class and run our fitting function, we'll have the weights for both models saved in one object.

We'll throw the metrics information into our class as well:


```python
#Results are good (mostly because we're using dummy data), but we definitely want to save the model for later use. 
# 
# Create a class so all models can be combined into one and saved in a serialized pickle file: 

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
        

```

Looks good! Now let's create a new instance of our class and then save it as a serialized file for later use: 


```python
audit_cat_X_train= X_train['COMP_OFFICE_SERVICE']
audit_cat_y_train = y_train['AUDIT_CAT']
team_X_train = X_train[['COMP_OFFICE_SERVICE', 'REGION']]
team_y_train = y_train['TEAM']

pipes_model = Basement(pipeline_audit_cat,pipeline_team) #Instantiate the class, these will be pipeline_1 and pipline_2 in the class we created above
pipes_model.fit_pipes(audit_cat_X_train,audit_cat_y_train, team_X_train,team_y_train)
predictions = pipes_model.predict_pipes(X_test)
print(predictions)
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
      <th>DEPT_NUM</th>
      <th>DEPT_NAME</th>
      <th>COMP_OFFICE_SERVICE</th>
      <th>REG_PROGRAM</th>
      <th>REGION</th>
      <th>AUDIT_CAT_x</th>
      <th>TEAM_x</th>
      <th>AUDIT_CAT_y</th>
      <th>TEAM_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Human_Resources_1</td>
      <td>ABC_Corp.Headquarters.Employee_Onboarding</td>
      <td>North_America.Onboarding_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Finance_1</td>
      <td>ABC_Corp.Headquarters.Accounts_Payable</td>
      <td>Europe.Financial_Management</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Information_Technology_1</td>
      <td>ABC_Corp.Tech_Hub.Technical_Support</td>
      <td>Asia.Tech_Support_Program</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Marketing_1</td>
      <td>ABC_Corp.Marketing_Office.Advertising</td>
      <td>South_America.Ad_Campaigns</td>
      <td>REGION_2</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Sales_1</td>
      <td>ABC_Corp.Sales_Department.Customer_Relations</td>
      <td>Africa.Customer_Engagement</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Research_and_Development_1</td>
      <td>ABC_Corp.R&amp;D_Center.Product_Development</td>
      <td>Australia.Innovation_Program</td>
      <td>REGION_3</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Customer_Service_1</td>
      <td>ABC_Corp.Call_Center.Call_Center_Operations</td>
      <td>North_America.Service_Excellence</td>
      <td>REGION_1</td>
      <td>CAT_A</td>
      <td>BLUE_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Legal_1</td>
      <td>ABC_Corp.Legal_Dept.Compliance</td>
      <td>Europe.Compliance_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Operations_1</td>
      <td>ABC_Corp.Operations_Office.Supplier_Management</td>
      <td>Asia.Supply_Chain_Management</td>
      <td>REGION_3</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_A</td>
      <td>BLUE_TEAM</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Logistics_1</td>
      <td>ABC_Corp.Warehouse.Warehouse_Management</td>
      <td>South_America.Logistics_Optimization</td>
      <td>REGION_2</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Human_Resources_2</td>
      <td>ABC_Corp.Headquarters.Employee_Onboarding</td>
      <td>North_America.Onboarding_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Finance_2</td>
      <td>ABC_Corp.Headquarters.Accounts_Payable</td>
      <td>Europe.Financial_Management</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Information_Technology_2</td>
      <td>ABC_Corp.Tech_Hub.Technical_Support</td>
      <td>Asia.Tech_Support_Program</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Marketing_2</td>
      <td>ABC_Corp.Marketing_Office.Advertising</td>
      <td>South_America.Ad_Campaigns</td>
      <td>REGION_2</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Sales_2</td>
      <td>ABC_Corp.Sales_Department.Customer_Relations</td>
      <td>Africa.Customer_Engagement</td>
      <td>REGION_3</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Research_and_Development_2</td>
      <td>ABC_Corp.R&amp;D_Center.Product_Development</td>
      <td>Australia.Innovation_Program</td>
      <td>REGION_3</td>
      <td>CAT_A</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Customer_Service_2</td>
      <td>ABC_Corp.Call_Center.Call_Center_Operations</td>
      <td>North_America.Service_Excellence</td>
      <td>REGION_1</td>
      <td>CAT_A</td>
      <td>BLUE_TEAM</td>
      <td>CAT_A</td>
      <td>BLUE_TEAM</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Legal_2</td>
      <td>ABC_Corp.Legal_Dept.Compliance</td>
      <td>Europe.Compliance_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Operations_2</td>
      <td>ABC_Corp.Operations_Office.Supplier_Management</td>
      <td>Asia.Supply_Chain_Management</td>
      <td>REGION_3</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Logistics_2</td>
      <td>ABC_Corp.Warehouse.Warehouse_Management</td>
      <td>South_America.Logistics_Optimization</td>
      <td>REGION_2</td>
      <td>CAT_C</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Human_Resources_3</td>
      <td>ABC_Corp.Headquarters.Employee_Onboarding</td>
      <td>North_America.Onboarding_Program</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_C</td>
      <td>GREEN_TEAM</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Finance_3</td>
      <td>ABC_Corp.Headquarters.Accounts_Payable</td>
      <td>Europe.Financial_Management</td>
      <td>REGION_1</td>
      <td>CAT_B</td>
      <td>RED_TEAM</td>
      <td>CAT_A</td>
      <td>BLUE_TEAM</td>
    </tr>
  </tbody>
</table>
</div>




```python
joblib.dump(pipes_model, 'hierarchy_prediction_model.pkl') #serialize model weights for future use
```




    ['hierarchy_prediction_model.pkl']



Huzzah! Model saved!

Let's try reusing it. After all, your office wants you to map these departments every week. You're going to need to rerun your model successfully.

### Loading and Running with New Data



```python
# Now let's try it on some new (also dummy) data and export the results in a usable form, aka, Excel

#Loading the model back in to run on new data

loaded_model = joblib.load('hierarchy_prediction_model.pkl')

new_data = pd.DataFrame({
    'DEPT_NUM': ['112', '113', '114'],
    'DEPT_NAME': ['Legal_12','Marketing_12','Operations_12'],
    'COMP_OFFICE_SERVICE': ['ABC_Corp.Legal_Dept.Compliance', 'ABC_Corp.Marketing_Office.Advertising', 'ABC_Corp.Operations_Office.Supplier_Management'],
    'REG_PROGRAM': ['North_America.Onboarding_Program','Asia.Tech_Support_Program','Asia.Tech_Support_Program'],
    'REGION': ['REGION_1', 'REGION_3', 'REGION_3']
})

new_data.head()
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
      <th>DEPT_NUM</th>
      <th>DEPT_NAME</th>
      <th>COMP_OFFICE_SERVICE</th>
      <th>REG_PROGRAM</th>
      <th>REGION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>112</td>
      <td>Legal_12</td>
      <td>ABC_Corp.Legal_Dept.Compliance</td>
      <td>North_America.Onboarding_Program</td>
      <td>REGION_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113</td>
      <td>Marketing_12</td>
      <td>ABC_Corp.Marketing_Office.Advertising</td>
      <td>Asia.Tech_Support_Program</td>
      <td>REGION_3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>114</td>
      <td>Operations_12</td>
      <td>ABC_Corp.Operations_Office.Supplier_Management</td>
      <td>Asia.Tech_Support_Program</td>
      <td>REGION_3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Predict using the loaded model
predictions = loaded_model.predict_pipes(new_data)

# Print predictions
print(predictions)
```

      AUDIT_CAT      TEAM
    0     CAT_B  RED_TEAM
    1     CAT_A  RED_TEAM
    2     CAT_B  RED_TEAM


Okay! We've got our predictions. Now let's join them back to our new data.

### Save and Export


```python
results = loaded_model.rejoin(new_data,predictions)
print(results)
```

      DEPT_NUM      DEPT_NAME                             COMP_OFFICE_SERVICE  \
    0      112       Legal_12                  ABC_Corp.Legal_Dept.Compliance   
    1      113   Marketing_12           ABC_Corp.Marketing_Office.Advertising   
    2      114  Operations_12  ABC_Corp.Operations_Office.Supplier_Management   
    
                            REG_PROGRAM    REGION AUDIT_CAT      TEAM  
    0  North_America.Onboarding_Program  REGION_1     CAT_B  RED_TEAM  
    1         Asia.Tech_Support_Program  REGION_3     CAT_A  RED_TEAM  
    2         Asia.Tech_Support_Program  REGION_3     CAT_B  RED_TEAM  



```python
results.to_excel("results.xlsx") #exporting our results to Excel
```

I hope you found this exercise useful! It was a very simple dummy data set, but the mechanism can work for more complicated ones with more detailed logical rules. 

Please feel free to find me on LinkedIn www.linkedin.com/in/morgana-kinlan !

Happy coding :)


### BONUS - Deploy into Production using Flask

But wait there's more! 

One of the most important things to be able to do with a saved model is deploy it. One way to do this is by using Flask. In a nutshell, Flask allows you to build a web interface so that a user can send data to your model and receive the results. 

To see an example of how to do this on a development server with the project above, download this repot (make sure you keep the folder structure the same, because Flask looks for HTML files in a "templates" folder), and run "main.py". You'll see something like this in your terminal: 

![Screenshot 2025-01-27 at 3 45 12 PM](https://github.com/user-attachments/assets/bf60d7c3-1fb0-4ebb-b4c1-39687f8ad2d9)

Press [CMD + click] over the url to open the application on a local host, or just copy/paste it into a browser. This will bring you to the main interface, which is the code in "index.html" : 

![Screenshot 2025-01-27 at 3 52 34 PM](https://github.com/user-attachments/assets/12d027b7-0d5c-441d-a202-ec1f32c46eaf)

Click the "Choose File" button, and navigate to the file "new_data.xlsx". It's important to use this data for this exercise, because it is set up in the format that the model will expect. If you were to use this application in the real world, it would be a good idea to build-in some error handling so that the model will give an error if the new data isn't in the format expected. As written now, the app will crash if the user tries to upload data in the wrong format. This could also be handled by building in a form, so that the user can only enter in data in the specific format that the model requires. 

Click "Submit" to run the model. Once the model is run, the results (the predictions) will display on the screen, with an option for the user to download the results as a csv file. 

![Screenshot 2025-01-27 at 3 59 41 PM](https://github.com/user-attachments/assets/1085334a-8894-4223-a130-55466735a14d)

Voila!




