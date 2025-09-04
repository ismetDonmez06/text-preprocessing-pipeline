import re
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from nltk import download


class Preprocessor:
    def __init__(self, language="english", min_freq=0):
        # stopwords for chosen language
        self.stop_words = set(stopwords.words(language))
        # number of rare words to drop (default=5)
        self.min_freq = min_freq
        self.rare_words = set()

    def _clean_text(self, text):
        # lowercase
        text = text.lower()
        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # remove numbers
        text = re.sub(r"\d", "", text)
        # remove stopwords
        words = [w for w in text.split() if w not in self.stop_words]
        # lemmatize
        words = [Word(w).lemmatize() for w in words]
        return " ".join(words)

    def _find_rare_words(self, texts):
        # join all texts into one string
        all_texts = " ".join(texts)
        words = all_texts.split()
        word_counts = pd.Series(words).value_counts()

        # take N least frequent words (min_freq is adjustable)
        rare = word_counts.tail(self.min_freq)
        self.rare_words = set(rare.index)

    def basic_preprocess(self, texts):
        # just clean texts
        return [self._clean_text(t) for t in texts]

    def advanced_preprocess(self, texts):
        # clean texts
        cleaned = self.basic_preprocess(texts)
        # find rare words
        self._find_rare_words(cleaned)
        # remove rare words from each text
        new_texts = []
        for t in cleaned:
            words = [w for w in t.split() if w not in self.rare_words]
            new_texts.append(" ".join(words))
        return new_texts


if __name__ == "__main__":
    # download needed nltk data
    download("stopwords")
    download("wordnet")

    sample_texts = [
        "This FILM is very nice, I like it!!!",
        "No, not good... very bad :( 100% waste."
    ]

    # small corpus → use smaller min_freq
    p = Preprocessor(min_freq=0)

    print("Basic:", p.basic_preprocess(sample_texts))
    print("Advanced:", p.advanced_preprocess(sample_texts))
