# implementar bag of words
# implementar TF-IDF
# implementar Word2Vec
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# passar para a própria classe o modelo

# funcao que recebe o modelo, faz o fit transform e retorna um df

# classe que recebe um df e faz conversão para Bag of Words ou TdF
def sklearn_bow(df, col, n_gram=1):
    vectorizer = CountVectorizer(ngram_range=(n_gram, n_gram))
    X = vectorizer.fit_transform(df[col])
    df_bow_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    return df_bow_sklearn

def sklearn_tdf(df, col, n_gram=1, min_df=10):
    vectorizer = TfidfVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(df[col])
    df_tdf_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    return df_tdf_sklearn


# definir classes com base em cada dimensão da entrega

# redução de dimensionalidade

# prints