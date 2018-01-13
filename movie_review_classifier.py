from nltk.corpus import movie_reviews
from random import shuffle
import nltk

def labeled_movie_reviews():

# labeled_movie_reviews = []

    for fileid in movie_reviews.fileids():
        if "neg" in fileid:
            yield (movie_reviews.words(fileid), "negative")
        elif "pos" in fileid:
            yield (movie_reviews.words(fileid), "positive")

# print(labeled_movie_reviews)
# shuffle(labeled_movie_reviews)

print("processing dictionary")
counter_dictionary = {}

for words, label in labeled_movie_reviews():
    for word in words:
        if word in counter_dictionary:
            counter_dictionary[word] += 1
        else:
            counter_dictionary[word] = 1

list_counter_dictionary = [(key,value) for key, value in counter_dictionary.items()]

sorted_counter = sorted(list_counter_dictionary, key=lambda x: -x[1])

print("filtering")
filtered = sorted_counter[107:8817]

filtered_words = [word for word, count in filtered]

def feature_extractor(words):
    feature = {}
    for word in words:
        if word in filtered_words:
            if word in feature:
                feature[word] += 1
            else:
                feature[word] = 1

    return feature


print("feature extraction")
featureset = [(feature_extractor(words), label)for words, label in labeled_movie_reviews()]

training_set = featureset[200:]
test_set = featureset[:200]

print("classification")
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("evaluation")
results = nltk.classify(classifier, test_set)
print(results)
