import gensim.models.keyedvectors as word2vec
import bertScore
from pyvi import ViTokenizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
def modelKMeans (input,typeModel,k):
    print(input)
    print(k)
    print(typeModel)
    import gensim 
    import gensim.models.keyedvectors as word2vec
    w2v_model = word2vec.KeyedVectors.load('w2v.model')
    if(typeModel =="1"):
        w2v_model = word2vec.KeyedVectors.load('w2v.model')
        print("default skip gram :")
    print("test")
    vocabulary = []
    for word in w2v_model.key_to_index.keys():
        vocabulary.append(word)
    print(len(vocabulary))

    contents_parsed = input.lower() 
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip() 

    import nltk
    sentences = nltk.sent_tokenize(contents_parsed)

    i = 0
    for sentence in sentences:
        print(i,"===", sentence)
        i += 1
    X = []
    for sentence in sentences:
        # sentence = gensim.utils.simple_preprocess(sentence)
        # sentence = ' '.join(sentence)
        sentence_tokenized = ViTokenizer.tokenize(sentence)
        # print(sentence_tokenized)
        words = sentence_tokenized.split(" ")
        sentence_vec = np.zeros((128))
        for word in words:
            if word in vocabulary:
                sentence_vec+=w2v_model[word]
        X.append(sentence_vec)
    n_clusters = getFindParametersK(len(X),k)
    print("Số k: ", n_clusters)
    for n in n_clusters:
        n = int(n)
        print(n)
        kmeans = KMeans(n_clusters=n)
        kmeans = kmeans.fit(X)

        print("K-Means Clustering\n")
        avg = []
        for j in range(n):
            print("Cụm", j+1)
            idx = np.where(kmeans.labels_ == j)[0]
            print(idx)
            avg.append(np.mean(idx))
            print("Thứ tự trung bình: ", round(np.mean(idx), 2))
            print("="*115)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        print("Các câu gần", n, "tâm cụm nhất",closest)
        ordering = sorted(range(n), key=lambda k: avg[k])
        # print(ordering)
        print("\nKết quả tóm tắt:\n")
        # summary = ' '.join(["(Câu " + str(closest[idx]) + ") " + sentences[closest[idx]] for idx in ordering])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        list_summary = []
        list_summary.append(summary)
        list_score = []
        score = bertScore.bert_score_sumary(input,summary)
        list_score.append(score)
        max_value = max(list_score)
        index = list_score.index(max_value)
        summary_final = list_summary[index]
        print("Số cụm cho kết quả tốt nhất : ", n_clusters[index], " với score là : ", max_value)
    return(summary_final)

def getRarametersK(X):
    n_sentence = len(X)
    if n_sentence == 1:
        n_clusters = [1]
    else:
        start_clusters = ( int(n_sentence/2) - 2 ) if ( int(n_sentence/2) - 2 ) > 0 else int(n_sentence/2)
        end_clusters = ( int(n_sentence/2) +  2 ) if ( int(n_sentence/2) + 2 ) < n_sentence else int(n_sentence/2)
        if start_clusters == end_clusters:
            n_clusters=[int(n_sentence/2)]
        else:
            n_clusters=range(start_clusters, end_clusters, 1)
    return n_clusters

def getFindParametersK(X,k):
    if(int(k) != 0):
        return range(k,k+1)
    else:
        n_sentence = X
        if n_sentence == 1:
            return range(1,2)
        elif ( n_sentence > 1 and n_sentence <= 5):
            return range(2,4)
        else:
            mean = round(n_sentence/2)
            return range(mean-1,mean+1) if (n_sentence >= 5 and n_sentence < 8) > 0 else range(mean-2,mean+2)