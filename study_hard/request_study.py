import requests 
URL = 'http://www.tistory.com' 
response = requests.get(URL) 
print(response.status_code)
print(response.text)

