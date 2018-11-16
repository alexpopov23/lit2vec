import gensim
import numpy
import os

def load_model(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False, datatype=numpy.float32)
    return model

def calc_centroid(words, model):
    centroid = None
    for word in words:
        if centroid is None:
            centroid = model[word]
        else:
            centroid += model[word]
    centroid /= len(words)
    return centroid

def most_similar(model, positive, negative, number):
    return model.most_similar(positive=positive, negative=negative, topn=number)

def cosine_similarity(a, b):
    return numpy.dot(a, b)/(numpy.linalg.norm(a)* numpy.linalg.norm(b))

if __name__ == "__main__":
    path_model = "/home/lenovo/dev/DeepReading/sf_vectors.txt"
    model_name = "SF_Model"
    output_dir = "/home/lenovo/dev/DeepReading/analyses/"
    # sentence = "merry little surge electricity pipe automatic alarm mood organ bed awaken person"
    phrase1 = ["merry", "little", "surge", "electricity"]
    phrase2 = ["pipe", "automatic", "alarm", "mood", "organ", "bed"]
    phrase3 = phrase1 + phrase2
    sent = phrase3 + ["awaken", "person"]
    phrase4 = ["surge", "electricity", "alarm", "mood", "organ"]
    phrase5 = ["mood", "electricity", "organ"]
    model = load_model(path_model)
    most_sim1 = most_similar(model, phrase1, [], 100)
    most_sim2 = most_similar(model, phrase2, [], 100)
    most_sim3 = most_similar(model, phrase3, [], 100)
    most_sim4 = most_similar(model, phrase4, [], 100)
    most_sim5 = most_similar(model, phrase5, [], 100)
    most_sim_sent = most_similar(model, sent, [], 100)
    sf_terms = ["cyborg", "android", "programming", "free will", "consciousness", "empathy", "telepathy", "cybernetic"]
    set = (most_sim1, phrase1), (most_sim2, phrase2), (most_sim3, phrase3), (most_sim4, phrase4), (most_sim5, phrase5),\
          (most_sim_sent, sent)
    for data, phrase in set:
        # with open(os.path.join(output_dir, model_name + "-" + "_".join(phrase)), "w") as out:
        #     to_write = ""
        #     for tuple in data:
        #         to_write += tuple[0] + "\t" + str(tuple[1]) + "\n"
        #     out.write(to_write.encode('utf-8'))
        with open(os.path.join(output_dir, model_name + "-SFTerms"), "w") as out:
            to_write = ""
            for sf_term in sf_terms:
                to_write += sf_term + " similarities: " + "\n\n"
                if len(sf_term.split()) > 1:
                    sf_term_vector = calc_centroid(sf_term.split(), model)
                else:
                    sf_term_vector = model[sf_term]
                for _, phrase in set:
                    combo_vector = calc_centroid(phrase, model)
                    sim = cosine_similarity(combo_vector, sf_term_vector)
                    to_write += sf_term + "\t" + "_".join(phrase) + "\t" + str(sim) + "\n"
            out.write(to_write.encode('utf-8'))
