import pandas as pd
import numpy as np
import os
import re
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


#чтение (хз надо ли)
def read_data(path):
    marketing_dealer = pd.read_csv(path , sep = ';')
    marketing_dealerprice = pd.read_csv(path , sep = ';')
    marketing_product= pd.read_csv(path, sep = ';', index_col= 0)
    marketing_product = marketing_product.dropna(subset='name')
    marketing_productdealerkey = pd.read_csv(path, engine='python', sep = ';')
    return marketing_dealer, marketing_dealerprice, marketing_product, marketing_productdealerkey

# обработка текста
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # отделение английских слов
    pattern = re.compile(r'(?<=[а-яА-Я])(?=[A-Z])|(?<=[a-zA-Z])(?=[а-яА-Я])')
    text = re.sub(pattern, ' ', text)
    # приведение к нижнему регистру 
    text = text.lower()
    # удаление символов
    text = re.sub(r'\W', ' ', str(text))
    # удаление одиноко стоящих слов
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # соотношения объемов 
    pattern2 = re.compile(r'\b\d+:\d+\s*-\s*\d+:\d+\b|\s*\d+:\d+\s*')
    text = re.sub(pattern2, ' ', text)
    return "".join(lemmatizer.lemmatize(text)) 


# векторизация названий
def vectoriz(marketing_dealerprice, marketing_product):
    df_1 = marketing_dealerprice[['product_name_lem']]
    df_1 = df_1.rename(columns={'product_name': 'name'})
    df_2 = marketing_product[['name_lem']]
    df_2 = df_1.rename(columns={'name_lem': 'name'})
    df = pd.concat([df_1, df_2])
    count_tf_idf = TfidfVectorizer()
    df = count_tf_idf.fit_transform(df['name'])
    df_1 = count_tf_idf.transform(df_1['name'])
    df_2 = count_tf_idf.transform(df_2['name'])
    df_1 = df_1.toarray()
    df_2 = df_2.toarray()
    return df_1, df_2

# получение матрицы с расстояниями
def matching_names(marketing_product, marketing_dealerprice, df_1, df_2):
    df = pd.DataFrame(index = marketing_product['id'], 
                    columns = marketing_dealerprice['product_key']+ '_' + pd.Series(range(marketing_dealerprice.shape[0])).astype(str), 
                    data = pairwise_distances(df_2, df_1, metric = 'cosine'))
    return df

# вывод n-го количества семантически похожих названий
def top_k_names(df, name, top_k):
    product_key = marketing_dealerprice[f'{name}']
    print(df.iloc[[product_key]].sort_values()[:top_k])



marketing_dealer, marketing_dealerprice, marketing_product, marketing_productdealerkey = read_data(path)
marketing_dealerprice['product_name_lem'] = marketing_dealerprice['product_name'].apply(lemmatize_text)
marketing_product['name_lem'] = marketing_product['name'].apply(lemmatize_text)
df_1, df_2 = vectoriz(marketing_dealerprice, marketing_product)
df = matching_names(marketing_product, marketing_dealerprice, df_1, df_2)




