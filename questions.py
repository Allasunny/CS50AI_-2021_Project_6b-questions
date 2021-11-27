import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)
    
    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

    
def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_in_folder = os.listdir(directory)
    content = dict()
    
    for filename in files_in_folder:
        file_path = os.path.join(directory, str(filename))
         
        with open(file_path) as f:
            s = f.read()
        content[filename] = s
        
    return content    

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
            
    contents = [
                    word.lower() for word in
                    nltk.word_tokenize(document)
                    if (word.isalpha() and word not in string.punctuation and word.lower() not in nltk.corpus.stopwords.words("english"))
                ]    
           
    return contents    
    
    
def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    idfs = dict()
    words = set()
    
    # get all words in documents
    for filename in documents:
        words.update(documents[filename])
        
    
    for word in words:
        # number of the same word containing in files
        f = sum(word in documents[filename] for filename in documents)
        
        # inverse document frequency 
        idf = math.log(len(documents) / f) 
        idfs[word] = idf
       
    return idfs
    
def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    tfidfs = dict()
    
    for filename in files:
        sum_tf_idf = 0
        for word in query:
            if word in files[filename]:
                
                # count number of times a term appears in a file
                tf = files[filename].count(word)
            
                # the sum of tf-idf values for any word in the query that also appears in the file
                sum_tf_idf += tf * idfs[word]
                
        # mapping filenames to tf-idf values by  words of query         
        tfidfs[filename] =  sum_tf_idf
   
    tfidfs_sorted = dict(sorted(tfidfs.items(), key=lambda x: x[1], reverse=True))
    tfidfs_list = [key for key in tfidfs_sorted]
   
    return tfidfs_list[:n]
    
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # the list of lists with sentences, idf values and query term density
    idfs_td = []
    
    for sentence in sentences:
        sum_idfs = 0
        score = 0  

        for word in query:
            if word in sentences[sentence]:
                
                # counts same words in sentence and query
                score += 1 
                
                # the sum of IDF values for any word in the query that also appears in the sentence
                sum_idfs += idfs[word]
        
        idfs_td.append([sentence, sum_idfs, score/len(sentences[sentence])])  
       
    # rank according to higher query term density    
    idfs_td_sorted_td = sorted(idfs_td, key=lambda x: x[2], reverse=True)
    
    # rank according to idf
    idfs_td_sorted_idfs = sorted(idfs_td_sorted_td, key=lambda x: x[1], reverse=True)
    
    # choose only 'n'- sentences - first values in lists
    top_sentences = [idfs_td_sorted_idfs[t][0] for t in range(n)]
    
    return top_sentences


if __name__ == "__main__":
    main() 
