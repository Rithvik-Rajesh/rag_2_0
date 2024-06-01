

import os
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import pdfplumber

PDF_PATH = 'data/pdf/'
SUMMARY_PATH = "data/summarize/"

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def summarize(content):
    text = ""

    chunk_size = 2000
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    iteration = 0
    for chunk in chunks:
        inputs = tokenizer([chunk],return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], max_length=700, early_stopping=False)
        text += [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids][0]
    
        iteration += 1
        print(f"{iteration = } |  {(iteration/len(chunks))*100}%")
    
    return text

def save_summary(content,filename):
    with open(f"{SUMMARY_PATH}{filename}.txt", "w") as f:
        f.write(content)
    print(f"Saved {filename}.txt")

def read_pdf(filename):
    text = ""
    with pdfplumber.open(f"{PDF_PATH}{filename}.pdf") as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def generate_summaries():
    filenames = [i.rstrip('.pdf') for i in os.listdir(PDF_PATH) if i.endswith(".pdf")]
    for filename in filenames:
        if filename == "doc1" or filename == "doc5":
            continue
        print(f"Generating summary for {filename}")
        content = read_pdf(filename)
        summary = summarize(content)
        save_summary(summary,filename)
    print("Done")

if __name__ == "__main__":
    generate_summaries()

