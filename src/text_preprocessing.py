import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from typing import List
import re


class TextPreprocessing:
    def __init__(self, df, column_name, new_column_name='text_preprocessed',
     language='portuguese') -> None:
        self.df = df
        self.column_name = column_name
        self.new_column_name = new_column_name
        self.language = language
        self.df[new_column_name] = self.df[column_name]
        # self.custom_stopwords = ['produto', 'pra']
    
    # pode ser que valha a pena manter as pontuações
    # outra opção seria usar o word_tokenizer
    def tokenize(self):
        self.df[self.new_column_name] = self.df[self.new_column_name].apply(
            self.apply_tokenizer) 

    # criar parâmetro para remover stopwords
    def remove_stopwords(self, use_nltk=True, language='portuguese', 
    custom_stopwords=['produto', 'pra']):        
        stopwords = self.load_stopwords(use_nltk, language, custom_stopwords)
        self.df[self.new_column_name] = self.df[self.new_column_name].apply(
            self.apply_stopwords_removal, stopwords=stopwords)

    def stemming(self):
        self.snowball = SnowballStemmer(language=self.language)

        self.df[self.new_column_name] = self.df[self.new_column_name].map(
            lambda x: [self.snowball.stem(y) for y in x]
            )
        
    def apply_stemming(self, tokenized_text):
        stemming_tokenized_text = [self.snowball.stem(y) for y in tokenized_text]
        return stemming_tokenized_text

    def remove_punctuation(self):
        self.df[self.new_column_name] = self.df[self.new_column_name].str.replace(
            pat='[^\w\s]', repl='', regex=True)

    def remove_accents(self):
        self.df[self.new_column_name] = self.df[self.new_column_name].str.normalize(
            'NFKD').str.encode(
            'ascii', errors='ignore').str.decode('utf-8')
    
    def pre_process(self, stemming=True):
        self.remove_punctuation()
        self.remove_accents()
        self.tokenize()
        self.remove_stopwords()
        if stemming:
            self.stemming()
        #  ajeitar isso aqui
        self.df['clean_text'] = self.df[self.new_column_name].apply(
            lambda x: ' '.join(x))
        return self.df


    @staticmethod
    def apply_tokenizer(text):
        # seleciona apenas letras e coloca todas em minúsculo
        # retorna a sentença como uma lista
        # if type(text)==str:
        letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', text.lower())
        # else:
        #     letras_min = np.nan
        return letras_min


    @staticmethod
    def apply_stopwords_removal(text: List[str], stopwords: List[str]):
        clean_text = [word for word in text if word not in stopwords]
        return clean_text

    @staticmethod
    # Adicionar opção de remover palavras do nltk
    # Expandir stopwords adicionais
    def load_stopwords(use_nltk=True, language='portuguese', custom_stopwords=['pra']):
        
        stopwords_list = []
        if use_nltk:
            # baixa as stopwords
            nltk.download('stopwords')

            # para escolher as stopwords do português adicionamos a opçaõ de língua "portuguese"
            stopwords_list = stopwords.words(language)
            print(stopwords)
        
        if custom_stopwords:
            stopwords_list = stopwords_list + custom_stopwords
            # Caso ocorram palavras repetidas
            stopwords_list = set(stopwords_list)

        return stopwords_list

    # funciona mal no português
    def lemmatize():
        pass