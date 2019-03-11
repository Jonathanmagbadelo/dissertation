from better_profanity import profanity
from pronouncing import  pronunciations

WORD_LIST = ["this", "is", "test", "yo", "fuck", "this", "shit", "nigga", "pussy", "bitch", "twerk", "ass", "dick", "hoes", "niggas"]


def suggest_words(clean, rhyme, words, context_word):
    words = filter_suggested_words(words) if clean else words
    words = filter_rhyme_words(context_word, words) if rhyme else words
    print(words)


def filter_suggested_words(words):
    return [word for word in words if not profanity.contains_profanity(word)]


def filter_rhyme_words(context_word, words):
    return words


suggest_words(True, True, WORD_LIST, "time")
