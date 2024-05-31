import requests
import json 


url = 'http://localhost:11434/api/chat'


def llama3(prompt):
    data = {
        "model" : "llama3:8b",
        "messages":[
            {
                "role":"user",
                "content":prompt
            }
        ],
        "stream" : False,
    }
    headers = {
        "Content-Type": "application/json"
    }

    response  = requests.post(url,headers=headers,json=data)
    return response.json()['message']['content']


def summarize(data):
    prompt = f"summarize the following paper detailly, focus on high entropy alloys and corrosion properties  \n\n\n {data}"
    return llama3(prompt)

# response = llama3("hi")
# print(response)