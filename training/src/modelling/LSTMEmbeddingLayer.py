class LSTMEmbeddingLayer:
    def __init__(self, inputs, word_to_id):
        self.inputs = inputs
        self.word_to_id = word_to_id
        self.vocab_size = len(word_to_id)
        self.id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))