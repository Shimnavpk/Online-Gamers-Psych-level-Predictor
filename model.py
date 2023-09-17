import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the CSV file
df = pd.read_csv("GamingStudy_data.csv", encoding='ISO-8859-1')

# Drop the columns we don't need
df = df.drop(['Game', 'Platform', 'earnings', 'League', 'Gender', 'Degree', 'Birthplace', 'Reference', 'Playstyle', 'accept', 'highestleague'], axis=1)
df = df.drop(['SPIN1', 'SPIN2', 'SPIN3', 'SPIN4', 'SPIN5',
              'SPIN6', 'SPIN7', 'SPIN8', 'SPIN9', 'SPIN10', 'SPIN11', 'SPIN12',
              'SPIN13', 'SPIN14', 'SPIN15', 'SPIN16', 'SPIN17', 'Narcissism',
              'Work', 'Residence', 'SPIN_T', 'Residence_ISO3', 'Birthplace_ISO3'], axis=1)

# Create a new column for anxiety level based on GAD_T scores
def get_anxiety_level(score):
    if score >= 0 and score <= 4:
        return "Minimal Anxiety"
    elif score >= 5 and score <= 9:
        return "Mild Anxiety"
    elif score >= 10 and score <= 14:
        return "Moderate Anxiety"
    elif score >= 15 and score <= 21:
        return "Severe Anxiety"
    
df['Anxiety_level'] = df['GAD_T'].apply(get_anxiety_level)

# Create a new column for life satisfaction level based on SWL_T scores
def get_life_satisfaction_level(score):
    if score >= 5 and score <= 9:
        return "Extremely dissatisfied"
    elif score >= 10 and score <= 14:
        return "Dissatisfied"
    elif score >= 15 and score <= 19:
        return "Slightly dissatisfied"
    elif score >= 20 and score <= 24:
        return "Slightly satisfied"
    elif score >= 25 and score <= 29:
        return "Satisfied"
    elif score >= 30 and score <= 35:
        return "Extremely satisfied"
    
df['Life_satisfaction'] = df['SWL_T'].apply(get_life_satisfaction_level)

# Drop the original GAD_T and SWL_T columns
df = df.drop(['S. No.', 'Timestamp', 'GAD1', 'GAD2', 'GAD3', 'GAD4', 'GAD5', 'GAD6','GAD7', 'GADE', 'SWL1', 'SWL2', 'SWL3', 'SWL4', 'SWL5','whyplay'], axis=1)
print(df.columns)
# Impute missing values using median imputation for numerical features and mode imputation for categorical features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


# Split the data into input features (X) and target variables (y)
X = df.drop(['Anxiety_level', 'Life_satisfaction'], axis=1)
y = df[['Anxiety_level', 'Life_satisfaction']]
print(X)
print(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a random forest classifier with multi-output support
rf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=8)
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)

# Save the trained model
pickle.dump(multi_rf, open('model.pkl', 'wb'))
