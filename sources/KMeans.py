def KMeans (content) :
    import gensim 
    import gensim.models.keyedvectors as word2vec
    w2v_model = word2vec.KeyedVectors.load('w2v.model')

    vocabulary = []
    for word in w2v_model.key_to_index.keys():
        vocabulary.append(word)
    print(len(vocabulary))

    contents_parsed = content.lower() 
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip() 

    import nltk
    sentences = nltk.sent_tokenize(contents_parsed)

    i = 0
    for sentence in sentences:
        print(i,"===", sentence)
        i += 1
    
    from pyvi import ViTokenizer
    import numpy as np 

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

    print(len(X))

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(X)

    print("K-Means Clustering\n")
    avg = []
    for j in range(n_clusters):
        print("Cụm", j+1)
        idx = np.where(kmeans.labels_ == j)[0]
        print(idx)
        avg.append(np.mean(idx))
        print("Thứ tự trung bình: ", round(np.mean(idx), 2))
        print("="*115)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    print("Các câu gần", n_clusters, "tâm cụm nhất",closest)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    # print(ordering)
    print("\nKết quả tóm tắt:\n")
    summary = ' '.join(["(Câu " + str(closest[idx]) + ") " + sentences[closest[idx]] for idx in ordering])
    print(summary)
    return(summary)