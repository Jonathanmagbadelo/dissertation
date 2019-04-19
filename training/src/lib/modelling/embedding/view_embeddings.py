from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def view_embedding(token_to_id, embeddings, word_count=200):
    plt.interactive(True)
    labels = []
    tokens = []

    for word, index in token_to_id.items():
        tokens.append(embeddings[index].tolist())
        labels.append(word)

    tokens, labels = tokens[:word_count], labels[:word_count]

    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    # Create dataframe
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'group': labels
    })

    p1 = sns.scatterplot(data=df, x="x", y="y", marker="o", hue='group', size=10)

    for line in range(0, df.shape[0]):
        p1.text(df.x[line] + 0.2, df.y[line], df.group[line], horizontalalignment='left', size=10, color='black',
                weight='semibold')

    plt.show()

    plt.savefig('embeddings/qvec-master/base_embedding.png')