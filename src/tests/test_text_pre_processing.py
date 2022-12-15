# testar classe que faz pré-processamento de texto
from text_preprocessing.text_preprocessing import TextPreprocessing
import pandas as pd
import pytest


class TestTextPreProcess():
    
    def test_tokenize(self):
        input = ['Texto separado por espaços, e com pontuações !, ? soltas. ',
                'Outro exemplo mais maluco !! :dd, .... .']
        df = pd.DataFrame(input, columns=['text'])

        text_preprocessor = TextPreprocessing(df, 'text', 'text_preprocessed')
        text_preprocessor.tokenize()

        expected = [['texto', 'separado', 'por', 'espaços', 'e', 'com', 'pontuações', 'soltas'],
                    ['outro', 'exemplo', 'mais', 'maluco', 'dd']]

        print(text_preprocessor.df['text_preprocessed'][0])
        assert text_preprocessor.df['text_preprocessed'][0] == expected[0]
        assert text_preprocessor.df['text_preprocessed'][1] == expected[1]
    

    def test_remove_ponctuation(self):
        input = ['Texto separado por espaços, e com pontuações !, ? soltas. ',
                'Outro exemplo mais maluco !! :dd, .... .']
        df = pd.DataFrame(input, columns=['text'])

        text_preprocessor = TextPreprocessing(df, 'text', 'text_preprocessed')
        text_preprocessor.remove_punctuation()

        expected = ['Texto separado por espaços e com pontuações   soltas ',
                    'Outro exemplo mais maluco  dd  ']

        assert df['text_preprocessed'][0] == expected[0]
        assert df['text_preprocessed'][1] == expected[1]

    def test_remove_accents(self):
        input = ['ábêlítuçáè !, ?', ' cõátírá']
        df = pd.DataFrame(input, columns=['text'])

        text_preprocessor = TextPreprocessing(df, 'text', 'text_preprocessed')
        text_preprocessor.remove_accents()

        expected = ['abelitucae !, ?', ' coatira']

        assert df['text_preprocessed'][0] == expected[0]
        assert df['text_preprocessed'][1] == expected[1]

    
# testar modelo que faz classificação com base no regex