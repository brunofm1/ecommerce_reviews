from config import __resources_dir__
import os
import numpy as np
from nltk.stem.snowball import SnowballStemmer
# import glob

# busca uma se a coluna alguma dentre uma lista de palavras
def column_regex(df, col_name, words, new_col_name):
    df[new_col_name] = False
    for word in words:
        df[new_col_name] = df[new_col_name] + (df[col_name].str.contains(word))
    return df


def read_one_word_line_txt(txt_path, dir=__resources_dir__):
    words = []
    with open(dir + txt_path) as f:
        # lines = f.readlines()
        for line in f:
            words.append(line.strip())
    return words


def list_stemming(words):
    snowball = SnowballStemmer(language='portuguese')
    stemm_words = set([snowball.stem(y) for y in words])
    return stemm_words


def regex_classification(df, dir=__resources_dir__, stemming=True, df_col='clean_text'):
    files = os.listdir(dir)
    col_name_value = {}
    for file in files:
        words = read_one_word_line_txt(file, dir)
        if stemming:
            words = list_stemming(words)

        name = file.strip('.txt')
        col_name = name + '_review'
        col_name_value[col_name] = name
        df = column_regex(df.copy(), df_col, words, col_name)

    aggregate_classes(df, col_name_value)

    return df

def aggregate_classes(df, columns_names_values:dict):
    df['review_class'] = ''

    for key, value in columns_names_values.items():
        df['review_class'] = np.where(df[key], df['review_class']+ '-' + value, 
                                        df['review_class'])

    # remover primeira letra (-)
    df['review_class'] = df['review_class'].str[1:]
    df['review_class'] = df['review_class'].replace({'': 'Nao identificado'})
    
       
