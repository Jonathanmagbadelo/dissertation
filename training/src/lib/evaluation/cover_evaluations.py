from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from src.lib.processing import utils
from scipy.spatial.distance import cosine
from statistics import mean


def most_similar_words(word, filename, top_n=10):
    glove_file = "embeddings/{}.txt".format(filename)
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    result = model.similar_by_word(word, topn=top_n)
    print(result)


def most_changed_words(base_embedding_filename, co_variate_embedding_filename):
    token_to_id, id_to_token, base_embeddings = utils.load_embeddings(base_embedding_filename)
    _, _, co_variate_embeddings = utils.load_embeddings(co_variate_embedding_filename)
    cosine_distances = {id_to_token[index]: cosine(np.array(base_embedding, dtype=np.float),
                                                   np.array(co_variate_embedding, dtype=np.float)) for
                        index, (base_embedding, co_variate_embedding) in
                        enumerate(zip(base_embeddings, co_variate_embeddings))}
    #s = [(k, cosine_distances[k]) for k in sorted(cosine_distances, key=cosine_distances.get, reverse=True)]
    print(mean(cosine_distances.values()))


most_similar_words("king", "cover_embeddings")
#most_changed_words("cover_embeddings", "cover_Hip Hop_embeddings")
