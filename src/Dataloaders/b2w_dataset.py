# Load do dataset da B2W e pré processamento básico dos dados
# com criação de novas colunas
import pandas as pd
from config import __data_dir__
import numpy as np

# idealmente vou herdar de uma classe pai que define as funcionalidades
class B2W:

    # obs: a coluna product_id tem os seguinte valor que impede ser lida como int: 20180316-4
    def __init__(self, data_dir=__data_dir__, path='b2w-reviews01-main/B2W-Reviews01.csv') -> None:
        self.df = pd.read_csv(data_dir+path, dtype={'product_id': str}, 
        parse_dates=['submission_date'])

    def compute_words_per_review(self):
        self.df['word_count'] = self.df[self.new_col].str.split().str.len()

    def drop_small_reviews(self, minimum_length=3):
        self.compute_words_per_review()
        original_len = len(self.df)
        self.df = self.df[self.df['word_count']>minimum_length].copy()
        new_len = len(self.df)
        print(f'{original_len-new_len} linhas deletadas por terem menos de {minimum_length} palavras')

        
    # parâmetro col - dúvida se deleto com base no full review ou apenas parte
    def drop_duplicates_reviews(self, col=False):
        original_len = len(self.df)
        if col:
            self.df = self.df.drop_duplicates(subset=[col])
        else:
            self.df = self.df.drop_duplicates(subset=[self.new_col])
        new_len = len(self.df)
        print(f'{original_len-new_len} linhas deletadas por terem reviews iguais')


    # dropar linhas sem nenhuma review e juntar título e texto
    def join_reviews(self, dropna=True, new_col='full_review'):  
        if dropna:
            self.df = self.df.dropna(subset=['review_text', 'review_title'], how='all').copy()

        self.new_col = new_col
        self.df[self.new_col] = self.df['review_title'].fillna('') + '. ' + self.df['review_text'].fillna('')


    def pre_process(self):
        self.join_reviews()
        self.drop_small_reviews()
        self.drop_duplicates_reviews()

    @staticmethod
    def words_count(review):
        if type(review)==str:
            return len(review.split())
        else:
            return np.nan
