import os

import gensim
import numpy
import nltk

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
from gensim.similarities.index import AnnoyIndexer

INPUT_STRING = "A merry little surge of electricity piped by automatic alarm from the mood organ beside his bed " \
               "awakened Rick Deckard. Surprised - it always surprised him to find himself awake without prior notice" \
               " - he rose from the bed, stood up in his multicolored pajamas, and stretched. Now, in her bed, his " \
               "wife Iran opened her gray, unmerry eyes, blinked, then groaned and shut her eyes again."
INPUT_STRING_NORMALIZED = "merry little surge electricity pipe automatic alarm mood organ beside bed awaken person" \
                          " surprised always surprise him find awake without prior notice rise bed stand up " \
                          "multicolored pajamas stretch now bed wife person open gray sad eyes blink groan shut" \
                          " eyes again"
INPUT_STRING_SENT1 = "merry little surge electricity pipe automatic alarm mood organ bed awaken person"
PHRASES = ["merry little surge electricity", "pipe", "automatic alarm mood organ bed", "awaken person"]
OUTPUT_FOLDER = "/home/lenovo/dev/DeepReading/analyses/"
MODEL = "GloVe"
if MODEL == "GloVe" or MODEL == "Combined":
    WORD_EMBEDDINGS_PATH = "/home/lenovo/dev/word-embeddings/glove.6B/glove.6B.300d_MOD.txt"
elif MODEL == "WordNet":
    WORD_EMBEDDINGS_PATH = "/home/lenovo/dev/word-embeddings/lemma_sense_embeddings/WN30WN30glConOne-C15I7S7N5_200M_syn_and_lemma_WikipediaLemmatized_FILTERED.txt"
elif MODEL == "SF":
    WORD_EMBEDDINGS_PATH = "/home/lenovo/dev/DeepReading/sf_vectors.txt"

wordnet_lexicon = "/home/lenovo/tools/ukb_wsd/lkb_sources/wn30.lex"
# get the mapping from synset to (first)lemma
synset2lemma = {}
with open(wordnet_lexicon, "r") as lexicon:
    for line in lexicon.readlines():
        line = line.strip()
        fields = line.split(" ")
        lemma, synsets = fields[0], fields[1:]
        for synset in synsets:
            if synset[:10] not in synset2lemma:
                synset2lemma[synset[:10]] = lemma
STOP_WORDS = nltk.corpus.stopwords.words()
model = gensim.models.KeyedVectors.load_word2vec_format(WORD_EMBEDDINGS_PATH, binary=False,
                                                                   datatype=numpy.float32)

def tsne_plot(model, words, syn2lemma=None):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in words:
        tokens.append(model[word])
        if syn2lemma is not None and word in syn2lemma:
            word = syn2lemma[word] + "%" + word
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# tsne_plot(model, data)

def get_similar_words(words, num_similar):
    sim_words = set()
    centroid = None
    for word in words:
        if centroid is None:
            centroid = model[word]
        else:
            centroid += model[word]
    centroid /= len(words)
    similar = model.most_similar(positive=[centroid], negative=[], topn=num_similar)
    for word in similar:
        if word in STOP_WORDS:
            similar.remove(word)
    sim_words.update(set(similar))
    # sim_words.update(words)
    return sim_words

# Get similar words per individual tokens in the sentence
words = set(INPUT_STRING_SENT1.split(" "))
for word in words:
    sim_words = get_similar_words([word], 100)
    with open(os.path.join(OUTPUT_FOLDER, word + "-" + MODEL + ".txt"),"w") as out:
        out.write(u"\n".join([sim_word[0] + "\t" + str(sim_word[1]) for sim_word in sim_words]).encode('utf-8').strip())

# Get similar words per phrase in the sentences
for phrase in PHRASES:
    sim_words = get_similar_words(phrase.split(" "), 100)
    with open(os.path.join(OUTPUT_FOLDER, phrase.replace(" ","_") + "-" + MODEL + ".txt"),"w") as out:
        out.write(u"\n".join([sim_word[0] + "\t" + str(sim_word[1]) for sim_word in sim_words]).encode('utf-8').strip())

# Get centroid for whole sentence and calculate neighbors:
sim_words = get_similar_words(words, 100)
with open(os.path.join(OUTPUT_FOLDER, "SENTENCE-" + MODEL + ".txt"), "w") as out:
    out.write(u"\n".join([sim_word[0] + "\t" + str(sim_word[1]) for sim_word in sim_words]).encode('utf-8').strip())


# words, scores = zip(*words_to_plot)
# wtp_filtered = []
# for i, word in enumerate(words):
#     if word not in STOP_WORDS:
#         wtp_filtered.append((word, scores[i]))
# tsne_plot(model, zip(*wtp_filtered)[0], synset2lemma)