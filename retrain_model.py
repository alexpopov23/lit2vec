from gensim.models.word2vec import Word2Vec, LineSentence
from copy import deepcopy

old_data = '/home/lenovo/dev/DeepReading/new_wackypedia_en_text.txt'
new_data = '/home/lenovo/dev/DeepReading/internet_archive_scifi_v3_tokenized.txt'
old_model = '/home/lenovo/dev/DeepReading/wiki_vectors.bin'
new_model = '/home/lenovo/dev/DeepReading/wikisf_vectors_alpha1.bin'

lines = LineSentence(old_data)
# model = Word2Vec(lines, size=100)
# oldmodel = deepcopy(model)
# oldmodel.save(old_model)
new_lines = LineSentence(new_data)
model = Word2Vec.load(old_model)
model.build_vocab(new_lines, update=True)
model.alpha = 0.1
model.train(new_lines)
model.save(new_model)

for m in ["oldmodel", "model"]:
    print('The vocabulary size of the', m, 'is', len(eval(m).wv.vocab))

print "This is the end"