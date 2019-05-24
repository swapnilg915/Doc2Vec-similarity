import gensim, os, collections, smart_open, random, time, json, re
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

def cleanText(doc):

    doc = doc.replace("<br>","")
    doc = doc.replace(":","")
    doc = doc.replace("</br>","")
    doc = doc.replace("<b>","")
    doc = doc.replace("</b>","")
    doc = doc.replace("?","")
    doc = doc.replace("/","")
    doc = doc.replace("\\","")
    doc = doc.replace("-","")
    doc = doc.replace("\/","")
    doc = doc.replace("\n","")
    doc = doc.replace(")","")
    doc = doc.replace("(","")
    doc = doc.lower().strip()
    doc = doc.encode("ascii", "ignore")
    return doc


text = "Zebras are several species of African equids (horse family) united by their distinctive black and white stripes. Their stripes come in different patterns, unique to each individual. They are generally social animals that live in small harems to large herds. Unlike their closest relatives, horses and donkeys, zebras have never been truly domesticated. There are three species of zebras: the plains zebra, the Grevy's zebra and the mountain zebra. The plains zebra and the mountain zebra belong to the subgenus Hippotigris, but Grevy's zebra is the sole species of subgenus Dolichohippus. The latter resembles an ass, to which it is closely related, while the former two are more horse-like. All three belong to the genus Equus, along with other living equids. The unique stripes of zebras make them one of the animals most familiar to people. They occur in a variety of habitats, such as grasslands, savannas, woodlands, thorny scrublands, mountains, and coastal hills. However, various anthropogenic factors have had a severe impact on zebra populations, in particular hunting for skins and habitat destruction. Grevy's zebra and the mountain zebra are endangered. While plains zebras are much more plentiful, one subspecies, the quagga, became extinct in the late 19th century - though there is currently a plan, called the Quagga Project, that aims to breed zebras that are phenotypically similar to the quagga in a process called breeding back."

documents = sent_tokenize(text)
print '\n documents --- ',documents

### data analysis
doc_vectors = count_vect.fit_transform(documents)
feature_names = count_vect.get_feature_names()
print "\n no. of unique words in data --- ",feature_names, len(feature_names)
print "\n vocab with index --- ",count_vect.vocabulary_
print "\n vocab with freq --- ",doc_vectors

###

dataset = [cleanText(sent) for sent in documents]
tagged_dataset = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]) for i, line in enumerate(dataset)]
dataset = tagged_dataset
print "\n len(dataset) === ",len(dataset), dataset[:3]

model = gensim.models.doc2vec.Doc2Vec(vector_size=20, min_count=2, epochs=30)
model.build_vocab(dataset)
model.train(dataset, total_examples = model.corpus_count, epochs=model.epochs)

### query testing
query = 'which zebras belongs to subgenus Hippotigris?'
query_tokens = query.split()
sent_vector = model.infer_vector(query_tokens)
print "\n query === ",query
print "\n model infer vector === ", sent_vector # len(sent_vector) = 50
sims = model.docvecs.most_similar([sent_vector], topn=3)
print "\n top 3 results sims === ",sims
sims_docs = [" ".join(dataset[tpl[0]].words) for tpl in sims]
print "\n similar doc === ",sims_docs