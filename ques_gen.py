
from model import llama3
import os
import json
import pdfplumber

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


PDF_PATH = 'data/pdf/'

filenames = [i for i in os.listdir(PDF_PATH) if i.endswith(".pdf")]


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def gen_dataset(pdf):
    # with fitz.open(pdf) as doc:
    #     res = ''
    #     for page in doc:
    #         res += page.get_text()

    res = extract_text_from_pdf(pdf)

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


    text = ""

    # Split the input text into chunks
    chunk_size = 2000
    stride = 1000
    chunks = [res[i:i+chunk_size] for i in range(0, len(res), stride)]
    
    summaries = []
    i = 0
    print(len(chunks))
    for chunk in chunks:
        inputs = tokenizer([chunk], return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], max_length=700, early_stopping=False)
        text += [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids][0]

        print("Loop "+str(i)+" Completed;")
        i+=1

    print(len(text))
    
    # # Combine the summaries into a single text
    # combined_summary = ' '.join(summaries)

    print(len(res))
    # print(len(combined_summary))


    with open("data/summarize/{pdf.strip("")}.txt",'w') as f:
        f.write(text)
        f.write("\n----------------------------------------------------\n")

    # prompt = f"""{res} \n\n\n create maximum number of questions with detail answer based on the above paper,
    # focus on Corrosion properties of high entropy alloys. 
    # questions regarding the publisher and publication details are not required. don't add any extra text. in list of json with question and answer format, and exclude 'Here are the questions and answers:' so that i can write directly to JSON file """

    # res = llama3(prompt)

    # return res


res = []

for pdf in filenames:
    res.append(gen_dataset(PDF_PATH + pdf))
    print(f"{pdf} completed")
    break

# print(res)

# with open("question_answer.json","w") as f:
#     f.write(res) 

