import re, itertools
import nltk
from nltk.corpus import stopwords
import csv
import string
from langdetect import detect
import multiprocessing

def csv2file(path):
    """
    Args: 
        path to csv file
    Output:
        list containing lines of csv file
    """
    with open(str(path), 'r') as f:
        content = csv.reader(f, delimiter = ',')
        contentlist = []
        for row in content:
            contentlist.append(row[0].strip())
    return contentlist


"""
pattern for to remove punctuation is defined below
"""
rmobject = string.punctuation
pattern = r'[{}]'.format(rmobject)

def remove_punct(string):
    """
    Args: 
        a string of characters (with punctuation)
    Output:
        a string of characters stripped from punctuation
    Dependencies:
        rmobject (above), pattern (above)
    """
    return re.sub(pattern, '', string)

def remove_url(string):
    """
    Args: 
        string
    Output:
        string contains 'http' T/F
    """
    if 'http:' in string:
        return True
    return False

def reducemultletter(string):
    """
    Args: 
        string of characters
    Output:
        string of characters without letter repetition (haaaaahaaa -> haahaa)
    """
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))

def clean_sentence(string):
    """
    Args: 
        string of characters
    Output:
        lowercased string without letter repetition and punctuation
    """
    tempsen = reducemultletter(string)
    tempsen = remove_punct(tempsen)
    return tempsen.lower()

def remove3wordsen(string):
    """
    Args: 
        string of characters
    Output:
        is the length of the string (#words) greater than 3: T/F 
    """
    if len(string.split()) <= 3:
        return True
    return False

def clean_data(data):
    """
    Args: 
        list of strings (e.g. from csv2file)
    Output:
        list of cleaned strings of wordcount >3
    """
    datanew = [clean_sentence(x) for x in data\
               if not remove3wordsen(clean_sentence(x))\
               and not remove_url(x)]
    return datanew

def avegsenlen(data):
    """
    Args: 
        list of strings
    Output:
        average length of strings (#words) in list
    """
    data = [x.split() for x in data]
    t = 0
    t = sum([len(x) for x in data])
    t = t/float(len(data))
    return t

def deldoubleword(data):
    """
    Args: 
        list of strings
    Output:
        list of strings without repeated words ('no more more please' -> 'no
        more please')
    """
    data = [x.split() for x in data]
    datalist = []
    for sen in data:
        sen2 = []
        for i in range(len(sen)):
            if sen[i-1] != sen[i]:
                sen2.append(sen[i])
        senstring = ''.join(x + ' ' for x in sen2).rstrip()
        datalist.append(senstring)
    return datalist

def fixhaha(data):
    """
    Args: 
        list of strings
    Output:
        list of strings where an attempt to fix the different variations of
        'haha' has been made.
    """
    data = [x.split() for x in data]
    datalist = []
    print 'begin haha fix'
    #pattern = re.compile('(h[ha]{2})')
    #pattern = re.compile('^[ha]+$|^[aha]+$')
    for i, sen in enumerate(data):
        if i % 100 == 0:
            print i
        for j, word in enumerate(sen):
            if 'hahaha' in word:
                data[i][j] = 'haha'
            elif 'hhaaha' in word:
                data[i][j] = 'haha'
            elif 'haaha' in word:
                data[i][j] = 'haha'
            elif 'hahaa' in word:
                data[i][j] = 'haha'
            elif 'hahha' in word:
                data[i][j] = 'haha'
        senstring = ''.join(x + ' ' for x in sen).rstrip()
        datalist.append(senstring)
    return datalist

def notnodkse(data):
    """
    Args: 
        list of strings
    Output:
        list of strings where an attempt to filter out strings of foreign languages
        (non-danish, swedish and norweigan) has been made.
    """
    print 'begin language filter'
    datalist = []
    languages = ['da', 'se', 'no']
    for i in range(len(data)):
        if i % 100 == 0:
            print i
        try: 
            if detect(unicode(data[i], 'utf8')) in languages:
                datalist.append(data[i])
        except:
            datalist.append(data[i])
    return datalist

if __name__ == '__main__':
    data = csv2file('data/ebdirty.csv')
    data = clean_data(data)
    data = deldoubleword(fixhaha(data))
    data = notnodkse(data)
    with open('data/cleaneb.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        for row in data:
            writer.writerow([row])
#verbatim = 'happpppy'
#verbatim = ''.join(''.join(s)[:2] for _, s in itertools.groupby(verbatim))
#print verbatim
