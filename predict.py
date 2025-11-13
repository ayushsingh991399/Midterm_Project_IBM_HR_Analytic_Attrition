import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Employee_Data(BaseModel): 
    age : int 
    businesstravel : str 
    dailyrate : int 
    department : str
    distancefromhome: int
    education  : str 
    educationfield : str
    environmentsatisfaction : str
    gender : str
    hourlyrate : int
    jobinvolvement : str
    joblevel : str
    jobrole : str
    jobsatisfaction : str
    maritalstatus : str
    monthlyincome : int
    monthlyrate : int
    numcompaniesworked : int
    overtime : str
    percentsalaryhike : int
    performancerating : str
    relationshipsatisfaction : str
    stockoptionlevel : int 
    totalworkingyears : int
    trainingtimeslastyear : int
    worklifebalance : str
    yearsatcompany : int
    yearsincurrentrole : int
    yearssincelastpromotion : int
    yearswithcurrmanager : int

with open("xgboost_model.bin", "rb") as f_in:
    model = pickle.load(f_in)

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)


app = FastAPI()
@app.get("/")
def home():
    return {"Hi you are on right path"}

@app.post("/predict")
def predict(client : Employee_Data ):
    
    dict = client.dict()
    X = dv.transform([dict])
    pred = model.predict(X) 
    label = int(pred[0])       

    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            prob = float(proba[0, 1]) 
        else:
            prob = float(proba[0])
    else:
        prob = None

    if label == 1:
        message = "Employee will LEAVE the company."
    else:
        message = "Employee will NOT leave the company."

    return {
        "prediction": label,
        "probability_of_leaving": prob,
        "message": message,
    }
