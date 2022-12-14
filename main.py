from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from models.regex_review_classification import regex_classification
from src.Dataloaders.b2w_dataset import B2W
from text_preprocessing.text_preprocessing import TextPreprocessing
from config import __file_dir__


if __name__=='__main__':

    # Load dataset
    b2w_dataset = B2W()
    b2w_dataset.pre_process()
    df = b2w_dataset.df.sample(frac=0.1)

    # pre process text
    text_preprocessor = TextPreprocessing(df, 'full_review')
    df = text_preprocessor.pre_process()

    # Review classification based on Regex
    df = regex_classification(df.copy())

    # Tdf word representation
    vectorizer_min = TfidfVectorizer(min_df=0.005)
    corpus = vectorizer_min.fit_transform(df["clean_text"])
    vocabulary_min = vectorizer_min.get_feature_names_out()

    # UMAP dimensio reduction
    reducer = umap.UMAP()
    x_umap = reducer.fit_transform(corpus)
    df["umpa_x_100"] = list(x_umap[:,0])
    df["umpa_y_100"] = list(x_umap[:,1])

    # plot
    df_plot = df[df['review_class'].isin(df['review_class'].value_counts().index[:6])]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x="umpa_x_100", y="umpa_y_100", hue="review_class", data=df_plot, 
                linewidth=0, ax=ax, s=4, alpha=.5, legend='full')
    plt.savefig('umap_visualization.png')