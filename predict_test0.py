import requests

url = 'http://localhost:3000/predict'

employee ={
  "age": 49,
  "businesstravel": "Travel_Frequently",
  "dailyrate": 279,
  "department": "Research & Development",
  "distancefromhome": 8,
  "education": "Below College",
  "educationfield": "Life Sciences",
  "environmentsatisfaction": "High",
  "gender": "Male",
  "hourlyrate": 61,
  "jobinvolvement": "Medium",
  "joblevel": "Junior Level",
  "jobrole": "Research Scientist",
  "jobsatisfaction": "Medium",
  "maritalstatus": "Married",
  "monthlyincome": 5130,
  "monthlyrate": 24907,
  "numcompaniesworked": 1,
  "overtime": "No",
  "percentsalaryhike": 23,
  "performancerating": "Outstanding",
  "relationshipsatisfaction": "Very High",
  "stockoptionlevel": 1,
  "totalworkingyears": 10,
  "trainingtimeslastyear": 3,
  "worklifebalance": "Better",
  "yearsatcompany": 10,
  "yearsincurrentrole": 7,
  "yearssincelastpromotion": 1,
  "yearswithcurrmanager": 7
}

response = requests.post(url, json=employee)

print(response.json())