import csv


def save_embedding(base_embeddings, token_to_id, filename):
    with open("embeddings/{}.txt".format(filename), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for word, index in token_to_id.items():
            word_embedding = base_embeddings[index].tolist()
            writer.writerow([word] + word_embedding)


def save_co_variate_embeddings(base_embeddings, co_variates, id_to_co_variate, token_to_id):
    for index, co_variate in enumerate(co_variates):
        co_variate_embeddings = base_embeddings * co_variate
        filename = "cover_{}_embeddings".format(id_to_co_variate[index])
        save_embedding(co_variate_embeddings, token_to_id, filename)


def save_co_variates(co_variates, filename):
    with open("embeddings/{}.txt".format(filename), "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for index, co_variate in enumerate(co_variates):
            co_variate_matrix = co_variate.tolist()
            writer.writerow([index] + co_variate_matrix)


def save_corpus(corpus, co_variates):
    with open("data/corpus.txt", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')
        for lyric, co_variate in zip(corpus, co_variates):
            writer.writerow([co_variate] + lyric)


def load_embeddings(filename):
    token_to_id = {}
    id_to_token = {}
    embeddings = []
    with open("embeddings/{}.txt".format(filename), "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for index, embedding in enumerate(reader):
            token = embedding[0]
            token_to_id[token] = index
            id_to_token[index] = token
            embeddings.append(embedding[1:])
    return token_to_id, id_to_token, embeddings


def load_co_variates(filename):
    with open("embeddings/{}.txt".format(filename), "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        co_variate_matrix = {}
        for co_variate in reader:
            co_variate_matrix[co_variate[0]] = co_variate[1:]
        return co_variate_matrix


def load_corpus():
    with open("data/corpus.txt", "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        corpus = []
        co_variates = []
        for document in reader:
            co_variates.append(document[0])
            corpus.append(document[1:])
        return co_variates, corpus
