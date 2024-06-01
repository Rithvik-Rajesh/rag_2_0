
from model import llama3
import os
import json

SUMMARIZED_PATH = 'data/summarize/'

filenames = [i for i in os.listdir(SUMMARIZED_PATH) if i.endswith(".txt")]


def gen_dataset(filename):

    with open(filename,'r') as f:
        content = f.read()

    # content = content[:3000]  

    chunk_size = 1000
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    iteration = 0
    with open('data/questions/data_doc6.json', 'a') as file:
        file.write('[\n')
        for chunk in chunks:
            prompt = f"""{chunk} \n\n\n create 8 questions with detail answer based on the above paper,
            focus on Corrosion properties of high entropy alloys. 
            questions regarding the publisher and publication details are not required. 
            don't add any extra text. in list of all the question and answer in json format, 
            give the answers in a little detailed manner,
            and exclude 'Here are the questions and answers:' so that i can write directly to JSON file """

            res = llama3(prompt)
            json_data = res.split('[')[1].split(']')[0]
            file.write(json_data)

            iteration += 1
            print(f"{iteration = } |  {(iteration/len(chunks))*100}%")

            if iteration < len(chunks):
                file.write(',')
            else:
                file.write('\n')

        file.write(']\n')

    # chunk_size = 1000
    # chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    # iteration = 0
    # for chunk in chunks:
    #     prompt = f"""{chunk} \n\n\n create 8 questions with detail answer based on the above paper,
    #     focus on Corrosion properties of high entropy alloys. 
    #     questions regarding the publisher and publication details are not required. 
    #     don't add any extra text. in list of all the question and answer in json format, 
    #     give the answers in a little detailed manner,
    #     and exclude 'Here are the questions and answers:' so that i can write directly to JSON file """

    #     res = llama3(prompt)
    #     result += res.split('[')[1].split(']')[0] + ","

    #     iteration += 1
    #     print(f"{iteration = } |  {(iteration/len(chunks))*100}%")

    # return result.rstrip(',')+"]"


res = []

for filename in filenames:
    gen_dataset(SUMMARIZED_PATH + filename)
    print(f"{filename} completed")
    break

# data = json.loads(result)
# with open('data.json', 'w') as file:
#     # Write the list of JSON objects to the file
#     json.dump(data, file, indent=2)

print("JSONL file created successfully!")