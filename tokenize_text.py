import nltk

f_in = "/home/lenovo/dev/DeepReading/internet_archive_scifi_v3.txt"
f_out = "/home/lenovo/dev/DeepReading/internet_archive_scifi_v3_tokenized.txt"
with open(f_in, "r") as infile:
    with open(f_out, "w") as outfile:
        text = infile.read()
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tokenized_sent = " ".join(tokens)
            outfile.write(tokenized_sent)
            outfile.write("\n")
