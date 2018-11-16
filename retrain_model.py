from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec, LineSentence

old_model = '/home/lenovo/dev/DeepReading/sf_vectors.bin'
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

print "This is the end"