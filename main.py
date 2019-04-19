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
F4_reweighting = dict()
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
            # print(df)
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
        #print("Doc=",doc_index,"    ",intersection)
        #print(innerProduct)
        sum = 0
        for i in innerProduct:
            sum = sum + i
        #print("sum = ", sum)
        query_length = vector_length(query)
        #print("query_length = ", query_length)
        doc_length = vector_length(tf_idf_table[doc_index])
        #print("doc_length = ", doc_length)
        ans = sum / (query_length*doc_length)
        #print("ans=",ans)
        cosSim_rate.append([doc_index+1, ans])
    return cosSim_rate


def vector_length(vector):
    sum = 0
    for i in vector:
        sum = sum + math.pow(i[1],2)
    ans = math.sqrt(sum)
    return ans

def Prob_of_Relevance(query):
    global f4
    N = number_of_doc
    R = len(relevant)
    n = dict()
    r = dict()
    #use relevant to find r and search all docs to find n.
    for term in query:
        if term[0] not in n:
            n[term[0]] = 0
            r[term[0]] = 0
        for doc_index in range(0,number_of_doc):
            if any(term[0] == x[0] for x in tf_table[doc_index]):
                n[term[0]] = n[term[0]] + 1
                if (doc_index) in relevant:
                    r[term[0]] = r[term[0]] + 1
    #print("n = ",n)
    #print("r = ",r)
    f4_dict = f4_measurement(query,N,n,R,r)
    reweighting(f4_dict)

def f4_measurement(query,N,n,R,r):
    F4 = dict()
    for term in query:
        dividend = (r[term[0]] + 0.5) / (R-r[ term[0] ] + 0.5)
        divisor = (0.5 + n[term[0]] - r[term[0]] ) / (0.5 + (N-n[term[0]]) - (R - r[term[0]]) )
        F4[term[0]] = math.log10(dividend/divisor)
    return F4


def reweighting(f4_dict):
    global F4_reweighting
    for doc_index in range(0,number_of_doc):
        f4_sum = 0
        for term in query:
            if any(term[0] == x[0] for x in tf_table[doc_index]):
                f4_sum = f4_sum + f4_dict[term[0]]
        F4_reweighting[doc_index] = f4_sum


read_common_words()
read_docs()
print_tf_table(tf_table)
print("\n----------------Print TF-IDF Table------------")
tf_idf()
print_tf_table(tf_idf_table)

print("\n----------------YOUR Query------------")
query = input("Enter Query:")
query = query_as_doc() #calculate Term Frequency for query
print("Query = ", query)

print("\n----------------Cosine Similaeiry------------")
similarity_rate = cosSim()
similarity_rate.sort(key=lambda x:x[1],reverse=True)
print("CosSim = ", similarity_rate)

print("\n----------------Probability of Relevance and F4 Measurement------------")
relevant = input("Enter Relevant docs' number with space:")
relevant = list(map(int,relevant.split()))  # change strings to int
relevant[:] = [x-1 for x in relevant]  # change range of numbers from 1:number_of_doc+1 to 0:number_of_doc
Prob_of_Relevance(query)
F4_reweighting = sorted(F4_reweighting.items(), key=lambda kv:(kv[1]), reverse=True)  # Sort docs based on F4 measure
F4_reweighting = [list(elem) for elem in F4_reweighting]
for element in F4_reweighting:  # change range of numbers to normal range
    element[0] = element[0] + 1
print(F4_reweighting)
