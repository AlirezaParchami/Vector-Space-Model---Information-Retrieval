from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import OrderedDict
import math
import string

frequent_words = set()
ps = PorterStemmer()
number_of_doc = 5
tf_table = []
tf_idf_table = []
def read_common_words():
    f = open("./Documents/frequent.txt", "r")
    for x in f:
        frequent_words.add(x.lower().rstrip('\n'))
    f.close()

def read_docs():
    for i in range(1, number_of_doc + 1):
        st = "./Documents/" + str(i) + ".txt"
        my_file = open(st, "r")
        f = my_file.read().lower()
        f = f.split()
        # We find all indices of a word in document and add in Inverted index
        # So I add the word in added_words[] to prevent appear repeated indices for same docID in inverted list.
        added_terms = []
        for word in f:
            # I should consider the main part of word
            edited_word = word.replace('.', '').replace('?', '').replace(',', '')
            stem_word = ps.stem(edited_word)
            # I use edited_word in condition because stem_word is wrong.
            # Ex. friends and friend are different words that I should find its indices but have same stem
            if stem_word not in frequent_words:
                is_existing = [item for item in added_terms if item[0] == stem_word]
                if len(is_existing) == 0:
                    # Entry_value is a dictionary of docID and Positions
                    added_terms.append([stem_word,1])
                if len(is_existing) != 0:
                    is_existing[0][1] = is_existing[0][1] + 1
        tf_table.append(added_terms)

def print_tf_table(x):
    for i in x:
        print(i)

def tf_idf():
    for doc in tf_table:
        doc_tf_idf = []
        for term in doc:
            df = len([item for row in tf_table for item in row if item[0] == term[0]])  # Document Frequency
            print(df)
            idf = math.log10(len(tf_table)/df)
            tfidf = term[1] * idf
            doc_tf_idf.append([term[0],tfidf])
        tf_idf_table.append(doc_tf_idf)

def query_as_doc():
    global query
    q = query.split()
    tf_q = list()
    for term in q:
        term = term.replace('.', '').replace('?', '').replace(',', '')
        term = ps.stem(term)
        # Calcuate Term Frequency
        is_existing = [item for item in tf_q if item[0] == term]
        if len(is_existing) == 0:
            tf_q.append([term,1])
        if len(is_existing) != 0:
            is_existing[0][1] = is_existing[0][1] + 1
    print("Term Frequenct:\n",tf_q)

    # Calculate Document Frequency
    for term in tf_q:
        df = len([item for row in tf_table for item in row if item[0] == term[0]]) + 1 # we regard query as a document. the first statement search df among documents. but we should add 1 because query is a document too
        idf = math.log10( (len(tf_table)+1) / df)
        term[1] = term[1] * idf
    print("Document Frequenct:\n", tf_q)
    return tf_q

def cosSim():
    global query
    global tf_idf_table
    cosSim_rate = []
    for doc_index in range(0,len(tf_idf_table)):
        intersection = []
        innerProduct = []
        for q in query:
            for item in tf_idf_table[doc_index]:
                if item[0] == q[0]:
                    intersection.append(item)
                    innerProduct.append(item[1]*q[1])
        print("Doc=",doc_index,"    ",intersection)
        print(innerProduct)
        sum = 0
        for i in innerProduct:
            sum = sum + i
        print("sum = ", sum)
        query_length = vector_length(query)
        print("query_length = ", query_length)
        doc_length = vector_length(tf_idf_table[doc_index])
        print("doc_length = ", doc_length)
        ans = sum / (query_length*doc_length)
        print("ans=",ans)
        cosSim_rate.append([doc_index+1, ans])
    return cosSim_rate


def vector_length(vector):
    sum = 0
    for i in vector:
        sum = sum + math.pow(i[1],2)
    ans = math.sqrt(sum)
    return ans

read_common_words()
read_docs()
print_tf_table(tf_table)
print("----------------------")
tf_idf()
print_tf_table(tf_idf_table)
query = input("Enter Query:")
query = query_as_doc()
print("-----------------")
similarity_rate = cosSim()
similarity_rate.sort(key=lambda x:x[1])
print(similarity_rate)
