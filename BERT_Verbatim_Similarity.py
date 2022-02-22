# Databricks notebook source
# MAGIC %md # BERT 
# MAGIC 
# MAGIC Bert - capability to embed the essence of words inside densely bound vectors. Dense vectors = each value inside the vector has a value and purpose for holding that value (contracting *sparse vectors*). 
# MAGIC 
# MAGIC For the BERT support, this will be a vector comprising 768 digits. Those 768 values have our mathematical representation of a particular token — which we can practice as contextual message embeddings.
# MAGIC 
# MAGIC Unit vector denoting each token (product by each encoder) is indeed watching tensor (768 by the number of tickets).
# MAGIC 
# MAGIC We can use these tensors and convert them to generate semantic designs of the input sequence. We can next take our similarity metrics and measure the corresponding similarity linking separate lines.
# MAGIC 
# MAGIC **The easiest and most regularly extracted tensor is the last_hidden_state tensor, conveniently yield by the BERT model.**
# MAGIC 
# MAGIC Of course, this is a moderately large tensor — at 512×768 — and we need a vector to implement our similarity measures.
# MAGIC 
# MAGIC **To do this, we require to turn our last_hidden_states tensor to a vector of 768 tensors.**

# COMMAND ----------

# MAGIC %pip install transformers

# COMMAND ----------

# MAGIC %md ## Method 1: Sentence Transformers
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/

# COMMAND ----------

# MAGIC %pip install sentence-transformers

# COMMAND ----------

#Write some lines to encode (sentences 0 and 2 are both ideltical):
sen = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
sen_embeddings = model.encode(sen)
sen_embeddings.shape

# COMMAND ----------

# MAGIC %md 4 sentence embeddings from the 4 sentences, each holding 768 values. 
# MAGIC Using these embeddings, discover the cosine similarity linking for each. 

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity
#let's calculate cosine similarity for sentence 0 to each of the other sentences:
cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
)

# COMMAND ----------

# MAGIC %md ## Method 2: Transformers & Pytorch
# MAGIC 
# MAGIC Before arriving at the second strategy, it is worth seeing that it does the identical thing as the above, but at one level more below.
# MAGIC 
# MAGIC We want to achieve our transformation to the last_hidden_state to produce the sentence embedding with this plan. For this, we work the mean pooling operation.
# MAGIC Additionally, before the mean pooling operation, we need to design last_hidden_state; here is the code for it:

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel
import torch
#nitialize our model and tokenizer:
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

###Tokenize the sentences like before:
sent = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]

# initialize dictionary: stores tokenized sentences
token = {'input_ids': [], 'attention_mask': []}
for sentence in sent:
    # encode each sentence, append to dictionary
    new_token = tokenizer.encode_plus(sentence, max_length=128,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    token['input_ids'].append(new_token['input_ids'][0])
    token['attention_mask'].append(new_token['attention_mask'][0])
# reformat list of tensors to single tensor
token['input_ids'] = torch.stack(token['input_ids'])
token['attention_mask'] = torch.stack(token['attention_mask']) #The attention mask is an optional argument used when batching sequences together. Which tokens do you attend to, which do not? Which ones are masked in training?

# COMMAND ----------

#Process tokens through model:
output = model(**token)
output.keys()

# COMMAND ----------

#The dense vector representations of text are contained within the outputs 'last_hidden_state' tensor
embeddings = output.last_hidden_state
embeddings

# COMMAND ----------

embeddings.shape

#again, 4 embeddings with 768 values, max length 128 char?

# COMMAND ----------

# MAGIC %md After writing our dense vector embeddings, we want to produce a *mean pooling operation* to form a single vector encoding, i.e., sentence embedding).
# MAGIC 
# MAGIC To achieve this mean pooling operation, we will require **multiplying all values in our embeddings tensor by its corresponding attention_mask value to neglect non-real tokens.**
# MAGIC 
# MAGIC #### Building The Vector
# MAGIC For us to transform our last_hidden_states tensor into our desired vector — we use a mean pooling method.
# MAGIC 
# MAGIC Each of these 512 tokens has separate 768 values. This pooling work will take the average of all token embeddings and consolidate them into a unique 768 vector space, producing a ‘sentence vector’.
# MAGIC 
# MAGIC At the very time, we can’t just exercise the mean activation as is. We lack to estimate null padding tokens (which we should not hold).

# COMMAND ----------

# To perform this operation, we first resize our attention_mask tensor:
att_mask = token['attention_mask']
att_mask.shape

# COMMAND ----------

mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
mask.shape

# COMMAND ----------

#mean pooling operation - multiplying all values in our embeddings tensor by its corresponding attention_mask value
mask_embeddings = embeddings * mask 
mask_embeddings.shape

# COMMAND ----------

#Then we sum the remained of the embeddings along axis 1:
summed = torch.sum(mask_embeddings, 1)
summed.shape #this has the same shape as sen_embeddings

# COMMAND ----------

#Then sum the number of values that must be given attention in each position of the tensor:
summed_mask = torch.clamp(mask.sum(1), min=1e-9)
summed_mask.shape

# COMMAND ----------

mean_pooled = summed / summed_mask
mean_pooled

# COMMAND ----------

# MAGIC %md Once we possess our dense vectors, we can compute the cosine similarity among each — which is the likewise logic we used previously:

# COMMAND ----------

from sklearn.metrics.pairwise import cosine_similarity
#Let's calculate cosine similarity for sentence 0:
# convert from PyTorch tensor to numpy array
mean_pooled = mean_pooled.detach().numpy()
# calculate
cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)

# COMMAND ----------

#print documents in order of cosine similarity
cosine_sim = cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)
sims = sorted(list(enumerate(cosine_sim[0])), key=lambda item: -item[1])
for doc_position, doc_score in sims:
    print(doc_score, sent[doc_position])

# COMMAND ----------

#have to convert text to vector form to input new text
text = "The jello-filled coffin stayed the same for three years." 

# COMMAND ----------

# MAGIC %md # Sentence Transformer on New Data

# COMMAND ----------

#read in new data
import pandas as pd
import numpy as np

df = pd.read_csv('/dbfs/FileStore/tables/Verbatims_about_the_survey_coming_up_too_soon__for_Michelle___July_15__2021_.csv')

#text preprocessing w nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re

#strip punctuation, lowercase letters, all symbols
def preprocessing(s):
    #remove punctuation, lowercase letters
    s = re.sub(r"[^a-zA-Z0-9\s]", "", str(s).lower())
    #tokenize
    s = nltk.word_tokenize(s)
    
    #remove stopwords (including Dell)
    stop = nltk.corpus.stopwords.words('english')
    stop.append('dell')
    s = [word for word in s if word not in stop]
    
#     #stem & lemmatize
#     ps = PorterStemmer()
#     s = [ps.stem(word) for word in s]
    
#     lemmatizer = WordNetLemmatizer()
#     s = [lemmatizer.lemmatize(word) for word in s]
    
    return s
    
#clean text
df['clean_text']=df['Improve Text'].apply(preprocessing)
df['clean_text'].head()

# COMMAND ----------

# survey_keywords2 = ['survey',
#  'ask',
#  'questionnair',
#  'question',
#  'pop',
#  'ask',
#  'stop',
#  'open',
#  'ask',
#  'finish']

# #if keyword is in survey_keywords, flag
# '''checking if a string has keywords from a list'''
# def find_from_list(x,l):
#     words = [word for word in l if word in x.split()]
#     return words

# '''check if two lists have any element in common and code dummy column'''
# def check_lists(list1, list2):
#     # using any function 
#     out = any(check in list1 for check in list2) 

#     # Checking condition 
#     if out: 
#         return 1 
#     else : 
#         return 0 

# df['survey_flag'] = df['clean_text'].apply(lambda x: check_lists(x, survey_keywords2)) #flag for if keyword appears
# #pos['survey_keywords'] = pos['spacy_clean_text'].apply(lambda x: find_from_list(x, survey_keywords2)) #capture keyword that appears

# COMMAND ----------

#remove words that appear only once
from collections import defaultdict

frequency = defaultdict(int)
for text in df['clean_text']:
    for token in text:
        frequency[token] += 1

texts = [
    ' '.join([token for token in text if frequency[token] > 1])
    for text in df['clean_text']
]

texts

# COMMAND ----------

#Write some lines to encode (sentences 0 and 2 are both ideltical):

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
sen_embeddings = model.encode(texts)
sen_embeddings.shape

# COMMAND ----------

#print documents in order of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(
    [sen_embeddings[0]],
    sen_embeddings[1:]
)
sims = sorted(list(enumerate(cosine_sim[0])), key=lambda item: -item[1])
for doc_position, doc_score in sims:
    print(doc_score, texts[doc_position])

# COMMAND ----------

#test cosine similarity to new text

sen = ['survey comes up before i had a chance  to look for what i came for. too early']
new_embedding = model.encode(sen)
new_embedding.shape

# COMMAND ----------

#print documents in order of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(
    [new_embedding[0]],
    sen_embeddings[0:]
)
sims = sorted(list(enumerate(cosine_sim[0])), key=lambda item: -item[1])
for doc_position, doc_score in sims:
  if doc_score > 0.5:
    print(doc_score, df['Improve Text'].iloc[doc_position])

# COMMAND ----------

# MAGIC %md #### Sentence Transformer on CSAT new data

# COMMAND ----------

from pyspark.sql.types import StringType, DateType,StructType, StructField
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.functions import concat, col, lit, lead, unix_timestamp, countDistinct
from pyspark.sql.window import Window
#import pandas as pd
import os.path
import IPython
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType, IntegerType,DoubleType
from pyspark import SparkContext,SparkConf
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col

# COMMAND ----------

df1 = spark.read.parquet('/mnt/sdslake/sds/lake/workspaces/eservices/CSAT_dataset/dbo.DS_CSAT_DnD_Consolidation.parquet').filter(col('Fiscal_Week')>='202101')
df1 = df1.filter(col('avgSat_Overall_Sat') > 6).filter(col('Improve_Text').isNotNull())
df1.count()

# COMMAND ----------

df = df1.toPandas()

# COMMAND ----------

#text preprocessing w nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re

#strip punctuation, lowercase letters, all symbols
def preprocessing(s):
    #remove punctuation, lowercase letters
    s = re.sub(r"[^a-zA-Z0-9\s]", "", str(s).lower())
    #tokenize
    s = nltk.word_tokenize(s)
    
    #remove stopwords (including Dell)
    stop = nltk.corpus.stopwords.words('english')
    stop.append('dell')
    s = [word for word in s if word not in stop]
    
#     #stem & lemmatize
#     ps = PorterStemmer()
#     s = [ps.stem(word) for word in s]
    
#     lemmatizer = WordNetLemmatizer()
#     s = [lemmatizer.lemmatize(word) for word in s]
    
    return s
    
#clean text
df['clean_text']=df['Improve_Text'].apply(preprocessing)
df['clean_text'].head()

# COMMAND ----------

# survey_keywords2 = ['survey','questionnaire','pop', 'question', 'ask', 'surveys', 'started', 'start', 'complete', 'poll']

# #if keyword is in survey_keywords, flag
# '''checking if a string has keywords from a list'''
# def find_from_list(x,l):
#     words = [word for word in l if word in x.split()]
#     return words

# '''check if two lists have any element in common and code dummy column'''
# def check_lists(list1, list2):
#     # using any function 
#     out = any(check in list1 for check in list2) 

#     # Checking condition 
#     if out: 
#         return 1 
#     else : 
#         return 0 

# df['survey_flag'] = df['clean_text'].apply(lambda x: check_lists(x, survey_keywords2)) #flag for if keyword appears
# df2 = df[df['survey_flag']==1]

df2 = df

# COMMAND ----------

#remove words that appear only once
from collections import defaultdict

frequency = defaultdict(int)
for text in df2['clean_text']:
    for token in text:
        frequency[token] += 1

texts = [
    ' '.join([token for token in text if frequency[token] > 1])
    for text in df2['clean_text']
]

texts

# COMMAND ----------

#Write some lines to encode (sentences 0 and 2 are both ideltical):

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
sen_embeddings = model.encode(texts)
sen_embeddings.shape

# COMMAND ----------

# #print documents in order of cosine similarity
# from sklearn.metrics.pairwise import cosine_similarity

# cosine_sim = cosine_similarity(
#     [sen_embeddings[0]],
#     sen_embeddings[1:]
# )
# sims = sorted(list(enumerate(cosine_sim[0])), key=lambda item: -item[1])
# for doc_position, doc_score in sims:
#     print(doc_score, df2['Improve_Text'].iloc[doc_position])

# COMMAND ----------

#test cosine similarity to new text

# sen = ['survey questionnaire popped up too early']
sen = ["support assist not working"] #to test it out like this, can't filter by keywords
new_embedding = model.encode(sen)
new_embedding.shape

# COMMAND ----------

#print documents in order of cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(
    [new_embedding[0]],
    sen_embeddings[0:]
)
sims = sorted(list(enumerate(cosine_sim[0])), key=lambda item: -item[1])
for doc_position, doc_score in sims:
  if doc_score > 0.5:
    print(doc_score, df2['Improve_Text'].iloc[doc_position])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Overall, works way better when we have the ability to do some sort of filtering on the text with keywords. Create a py file with 2 inputs?
# MAGIC 
# MAGIC 1. Keyword Category (multiple choice)
# MAGIC 2. Phrase to match similarity to
# MAGIC 3. Output = csv with verbatims ranked by similarity to a phrase
# MAGIC 
# MAGIC (backend data = verbatims output of csat ai engine (including keyword category))

# COMMAND ----------

sorted_sims_df = pd.DataFrame(sims, columns =['idx', 'Similarity Score']).set_index('idx')
surveycol = df2[[
     'ID',
     'Fiscal_Year',
     'Fiscal_Quarter',
     'Fiscal_Week',
     'Improve_Text',
    'avgSat_Overall_Sat']]
sims_df = sorted_sims_df.merge(surveycol.reset_index(), how='left',left_index=True, right_index=True)
sims_df

# COMMAND ----------


