import requests

url = 'http://localhost:3000/predict'

employee ={
  "age": 41,
  "businesstravel": "Travel_Rarely",
  "dailyrate": 1102,
  "department": "Sales",
  "distancefromhome": 1,
  "education": "College",
  "educationfield": "Life Sciences",
  "environmentsatisfaction": "Medium",
  "gender": "Female",
  "hourlyrate": 94,
  "jobinvolvement": "High",
  "joblevel": "Junior Level",
  "jobrole": "Sales Executive",
  "jobsatisfaction": "Very High",
  "maritalstatus": "Single",
  "monthlyincome": 5993,
  "monthlyrate": 19479,
  "numcompaniesworked": 8,
  "overtime": "Yes",
  "percentsalaryhike": 11,
  "performancerating": "Excellent",
  "relationshipsatisfaction": "Low",
  "stockoptionlevel": 0,
  "totalworkingyears": 8,
  "trainingtimeslastyear": 0,
  "worklifebalance": "Bad",
  "yearsatcompany": 6,
  "yearsincurrentrole": 4,
  "yearssincelastpromotion": 0,
  "yearswithcurrmanager": 5
}


response = requests.post(url, json=employee)

print(response.json())