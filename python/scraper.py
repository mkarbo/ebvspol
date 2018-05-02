from facebook import GraphAPI
import csv 
def get_comments(postid, token):
    """
    Args: 
        Takes a postid and API access token,
    
    Out:
        returns all comments from that post in list format, unformatted.
    """
    commentlist = []
    graph = GraphAPI(token)
    postObj = graph.get_all_connections(id=postid, connection_name = 'comments')
    for x in postObj:
        temp = [x[key] for key in ['message', 'created_time']]
        commentlist.append(temp)
    return commentlist


def get_anon_comments(postid, token):
    """
    Args: 
        Takes a postid and API access token
    
    Out:
    returns all comments from that post in list format, unformatted.
    """
    commentlist, taglist = [], []
    graph = GraphAPI(token)
    postObj = graph.get_all_connections(id=postid, \
                                        connection_name = 'comments',\
                                        fields = 'message_tags')
    temp = [x for x in postObj]
    i = 0
    for y in temp:
        if len(y) > 1:
            taglist.append([y['message_tags'][0]['name'], i])
        i += 1
    comments = get_comments(postid, token)
    for x in taglist:
        j = x[1]
        comments[j][0] = comments[j][0].replace(x[0], 'tagged person')
    return comments


def get_pageid(name, token):
    """
    Args: 
        Takes a pagename (e.g. 'politiken'), and an API token
    
    Out:
        returns pageid
    """
    graph = GraphAPI(token)
    return graph.get_object(name)['id']


def get_posts(name, token):
    """
    Args: 
        Takes a pagename and API token
    
    Out:
        Outputs all postids on the page
    """
    graph = GraphAPI(token)
    pageObj = graph.get_object(id = name, fields = 'id')
    pageid = pageObj['id']
    postObj = graph.get_all_connections(id = pageid, connection_name='posts')
    postlist = []
    for x in postObj:
        temp = [x[key] for key in ['id', 'created_time']]
        postlist.append(temp)
    return postlist

def print_comments(content, token, n=0):
    """
    Args: 
        takes a list of comments, a token, (optionally an integer)
    
    Out:
        prints all comments on all posts (or on the first n posts)
    """
    if n == 0:
        for i in range(len(content)):
            comments = get_comments(content[i][0], token)
            if len(comments) > 0:            
                for j in range(len(comments)):
                    print(comments[j][0])
    else:
        for i in range(n):
            comments = get_comments(content[i][0], token)
            if len(comments) > 0:            
                for j in range(len(comments)):
                    print(comments[j][0])


def save_comments(name, token, nfrom = 0, toappend=[]):
    """
    Args: 
        Takes a pagename, an API token, (optionally: an integer specifying
        which # post to begin from, and a list to append comments to)
    
    Out:
        a list containing:
            a list of of the comments it has iterated over, and an integer
            specifying how many posts it has iterated over.
    OBS:
        this often takes longer than the normal token lifetime. So specify
        a list to append to. If token runs out, you are prompted to enter a new
        token and it will print which post you came to. Then you can continue
        by specifying this as the optional integer above. Also attempts to save
        current progress in a temp.csv file.
    """
    postids = get_posts(name, token)
    k=1
    print('number of posts to scrape from: ' + str(len(postids)))
    for i in range(nfrom, len(postids)):
        try:
            comments = get_comments(postids[i][0], token)
        except:
            print('broke at:' + str(i))
            temp = toappend[:]
            temp = [[x] for x in temp]
            with open('temp.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter = ',')
                for j in len(temp):
                    writer.writerow(temp[j])
            print('temp file written')
            return([toappend, i])
        if len(comments) > 0:
            if i % 10 == 0:
                print('******************************************')
                print('post number: ' + str(i))
                print('first comment on post:')                      
                print(comments[0][0])
                print('comment number: ' + str(k))
                print('post date:' + str(comments[0][1]))
            for j in range(len(comments)):
                toappend.append(comments[j][0])
                k += 1
    return [toappend, i]


if __name__ == '__main__':
    pass
"""
I ran this script through termianl using import.
"""


