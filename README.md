# ecommerce_reviews
Processamento de Linguagem Natural reviews a partir de reviews de sites de e-commerce.

Este projeto tem como objetivo a implementação de técnicas de NLP em um contexto de review de produtos em sites de e-commerce. Dois datasets são trabalhados no projeto. São eles:

- B2W (https://github.com/americanas-tech/b2w-reviews01)
- Olist (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

O objetivo inicial do projeto foi a construção de um frame work para leitura e pré processamento dos datasets, assim como implementação de técnicas de representação de texto (Bag of Words e TDF). Além disso, foram implementadas técnicas não supervisionadas de clusterização e topificação para análise de dados e criado um modelo de classificação dos modelos com base em regex.


# main.py

Para rodar a partir da raiz do projeto basta digitar o comando dentro do ambiente virtual a ser gerado baixando os pacotes especificados dentro do arquivo requeriments.txt: 

python main.py

O dataset B2W será então carregado e processado, assim como as etapas de pré processamento de texto. Também será rodado o modelo de classificação das reviews a partir de um modelo baseline que faz topificação dos comentário com base em regex (). Em seguida, será o texto dos comentários será representado usando a técnica de Tdf () e será aplicada uma redução de dimensionalidade UMAP. Finalmenete, será gerada uma imagem da representação bidimensional da representação transformada.

Esse visualização, por sua vez, tem o objetivo de dar uma ideia inicial sobre a qualidade da representação dos comentários usando a técnica de Tdf. Tendo como base o modelo baseline (regex), a correta clusterização da classes indicariam ser sim uma boa representação adequada.

No entanto, há de se fazer uma série de ressalvas importantes. Em primeiro lugar, as classes definidas para as reviews são classes abstratas, não precisamente bem definidas e apenas algumas das diversas classes que podemos pensar em um contexto review de produtos. Não somente, o modelo utilizado para fazer a classificação é bastante superficial, não sendo esperado haver uma precisão grande. Por isso, usar a classificação do modelo como forma de avaliar a clusterização é uma forma de aproximação.
