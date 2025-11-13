from pydantic import BaseModel
from typing import Any, Dict
import bentoml
from bentoml.io import JSON


class Employee_Data(BaseModel):
    age: int
    businesstravel: str
    dailyrate: int
    department: str
    distancefromhome: int
    education: str
    educationfield: str
    environmentsatisfaction: str
    gender: str
    hourlyrate: int
    jobinvolvement: str
    joblevel: str
    jobrole: str
    jobsatisfaction: str
    maritalstatus: str
    monthlyincome: int
    monthlyrate: int
    numcompaniesworked: int
    overtime: str
    percentsalaryhike: int
    performancerating: str
    relationshipsatisfaction: str
    stockoptionlevel: int
    totalworkingyears: int
    trainingtimeslastyear: int
    worklifebalance: str
    yearsatcompany: int
    yearsincurrentrole: int
    yearssincelastpromotion: int
    yearswithcurrmanager: int

dictvec_ref = bentoml.sklearn.get("employee_dict_vectorizer:latest")
dictvec_runner = dictvec_ref.to_runner()
model_ref = bentoml.sklearn.get("employee_attrition_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service(
    "employee_attrition_service",
    runners=[dictvec_runner, model_runner],
)

@svc.api(input=JSON(pydantic_model=Employee_Data), output=JSON())
async def predict(employee: Employee_Data) -> Dict[str, Any]:

    input_dict = employee.dict()
 
    X = await dictvec_runner.transform.async_run([input_dict])

    pred = await model_runner.predict.async_run(X)
    pred_value = pred[0]

    
    proba = None
    if hasattr(model_ref, "predict_proba"):
        try:
            proba_arr = await model_runner.predict_proba.async_run(X)
            proba = proba_arr[0].tolist()
        except:
            pass

    return {
        "prediction": int(pred_value) if hasattr(pred_value, "item") else pred_value,
        "probability": proba,
        "raw_input": input_dict,
    }
