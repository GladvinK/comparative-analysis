import pandas as pd
import numpy as np
import re

#get data set
df = pd.read_csv("../resource/fake reviews dataset.csv")

df = df.replace('',np.nan,regex = True)

# clean data by removing sentence that dont end with proposition and punctuati
import spacy
nlp = spacy.load("en_core_web_sm")
docs = df['text_']
for i in docs:
    i = i.rstrip()
    doc = nlp(i)
    for sent in doc.sents:
        count = 0
        token = sent[-1]
        #filter sentence by checking if sentence end with punctuation.
        if token.pos_ == "ADP":
            count = count + 1
            
    if count == 0 :
        #check if sentence end with punctuations
        if not re.match('[?.!]$',i[-1]):
            df = df.replace(i,np.nan)

#drop Nan values
#new dataframe with cleaned data
newdf = df.dropna()
print("len: ",len(newdf))
#add new dataframe to a csv file
newdf.to_csv("../resource/newData.csv")