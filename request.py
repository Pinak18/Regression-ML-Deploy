import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'YearsExperience':2, 'InterviewScore':9})

print(r.json())