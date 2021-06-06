from nltk.corpus import stopwords
from spacy.lang.en import English

cachedStopWords = stopwords.words("english")


def remove_stop_words(text):
    nlp = English()
    my_doc = nlp(text)
    token_list = [token.text for token in my_doc]
    filtered_sentence = [word for word in token_list if not nlp.vocab[word].is_stop]
    return " ".join(filtered_sentence)
