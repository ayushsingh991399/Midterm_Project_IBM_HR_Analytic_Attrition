# Packages Part

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix,roc_curve,root_mean_squared_error
from xgboost import XGBClassifier
import pickle
import bentoml


# Data Preparation Part

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
def preprocess_data(df):
    print("Preprocessing the data...")
    df = df.fillna(method='ffill')
    df.columns = df.columns.str.lower()
    df = df.drop_duplicates().reset_index(drop=True)
    df["attrition"]  = df["attrition"].map({'Yes' : 1 , 'No' : 0 })
    df =  df.drop(['employeecount' , 'standardhours' , 'over18','employeenumber'], axis=1)
    df["education"] = df["education"].replace({1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"})
    df["environmentsatisfaction"] = df["environmentsatisfaction"].replace({1:"Low",2:"Medium",3:"High",4:"Very High"})
    df["jobinvolvement"] = df["jobinvolvement"].replace({1:"Low",2:"Medium",3:"High",4:"Very High"})
    df["joblevel"] = df["joblevel"].replace({1:"Entry Level",2:"Junior Level",3:"Mid Level",4:"Senior Level", 5:"Executive Level"})
    df["jobsatisfaction"] = df["jobsatisfaction"].replace({1:"Low",2:"Medium",3:"High",4:"Very High"})
    df["performancerating"] = df["performancerating"].replace({1:"Low",2:"Good",3:"Excellent",4:"Outstanding"})
    df["relationshipsatisfaction"] = df["relationshipsatisfaction"].replace({1:"Low",2:"Medium",3:"High",4:"Very High"})
    df["worklifebalance"] = df["worklifebalance"].replace({1:"Bad",2:"Good",3:"Better",4:"Best"})
    x = df.drop('attrition', axis=1)
    y = df['attrition']
    print("✅Data preprocessing completed.")
    return df, x, y

df, x, y = preprocess_data(df)
# Vectorization Partdv 

def split_and_vectorize(x, y):
    print("Splitting the data into training and testing sets...")
    X_train, X_test,Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42,stratify=y)
    dv = DictVectorizer(sparse=False)
    train_dicts = X_train.to_dict(orient='records')
    test_dicts = X_test.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)
    features = dv.get_feature_names_out().tolist()
    print("✅Data splitting and vectorization completed.")
    return X_train, X_test, Y_train, Y_test, features,dv

X_train, X_test, Y_train, Y_test, features,dv = split_and_vectorize(x, y) 
# Model Training Part

learning_rate=0.05
max_depth=4
n_estimators=200
objective='binary:logistic'
eval_metric='logloss'
use_label_encoder=False
random_state=42
subsample=0.8
colsample_bytree=0.8

def train_xgboost(X_train,Y_train,learning_rate, max_depth, n_estimators, objective, eval_metric, use_label_encoder, random_state, subsample, colsample_bytree):
    print("Training the XGBoost model...")
    
    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        objective=objective,
        eval_metric=eval_metric,
        use_label_encoder=use_label_encoder,
        random_state=random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )
    model.fit(X_train, Y_train)
    print("✅Model training completed.")
    return model


model = train_xgboost(X_train,Y_train,learning_rate, max_depth, n_estimators, objective, eval_metric, use_label_encoder, random_state, subsample, colsample_bytree)
# Evalution Part
def evaluate(model, X_train, X_test, Y_train, Y_test):
    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    Y_test_proba = model.predict_proba(X_test)[:, 1]
    print("✅TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(Y_train, Y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(Y_train, Y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(Y_train, Y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("✅TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(Y_test, Y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(Y_test, Y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(Y_test, Y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("\n")
    print("\n")
    print("\n=========ROC AUC SCORE============ ")
    auc = roc_auc_score(Y_test, Y_test_proba)
    print(f"ROC AUC SCORE: {auc:.4f}")

    print("✅Model evaluation completed.")


evaluate(model, X_train, X_test, Y_train, Y_test)
# Model Saving Part
def model_saving_pickle(model, dv,df, model_path='employee_attrition_model.bin', dv_path='employee_dict_vectorizer.bin',data_path='preprocessed_data.csv'):
    print("Saving the model and DictVectorizer...")
    df.to_csv(data_path, index=False)
    with open(model_path, 'wb') as f_out:
        pickle.dump(model, f_out)
    with open(dv_path, 'wb') as f_out:
        pickle.dump(dv, f_out)     
    print(f"✅ Model saved to {model_path}")
    print(f"✅ DictVectorizer saved to {dv_path}")
    print(f"✅ Cleaned dataset saved to {data_path}")


def model_saving_bentoml(model,dv,df, model_path='employee_attrition_model.bin', dv_path='employee_dict_vectorizer.bin',data_path='preprocessed_data.csv'):
    print("Saving the model and DictVectorizer...")
    df.to_csv(data_path, index=False)
    bentoml.sklearn.save_model(dv_path, dv)
    bentoml.sklearn.save_model(model_path, model)
model_saving_pickle(model,dv,df)
model_saving_bentoml(model,dv,df)

print("✅All tasks completed successfully.")