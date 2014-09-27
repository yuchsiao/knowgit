from flask import Flask, jsonify, render_template, request, render_template
import flask
import numpy as np
import scipy as sp
import urllib2
import json
import base64
import pymysql
import os
import codecs
import csv
import math
import cPickle as pickle
import time
import markdown
import datetime
import calendar

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Global parameters
filename_tag = 'static/tags100m.csv'
filename_stopword = 'static/stopword_mysql'
token = "token e27373ef1be7b1ca7713410f011167167f482134"
ratio_star_fork = 10
node_scaling = 10
edge_scaling = 0.3
thres_link = 0.7
n_edge_limit = 20
n_tag_on_display = 5
tag_scale = 8.0
readme_len = 350
# database_name = 'github_language'
database_name = 'github_gaussian'
# database_name = 'github_sqrtlog'
mu = math.log10(2113)
sigma = 1
flag_normalize_tag = False

app = Flask(__name__)

db = pymysql.connect(user="root", host="localhost", charset='utf8')


# load tags
def normalize_tag_list(tag_list):
    result = []
    for item in tag_list:
        x = item[0] - mu
        result.append((math.exp(-x*x/(2*sigma*sigma)), item[1]))
    return result


def normalize_tag_dict(tag_dict):
    result = {}
    for item in tag_dict:
        x = tag_dict[item][1] - mu
        result[item] = (tag_dict[item][0], math.exp(-x*x/(2*sigma*sigma)))
    return result


def prepare_tag():
    with db, codecs.open(filename_tag, 'rU', 'utf-8') as tag_in, \
            codecs.open(filename_stopword, 'rU', 'utf-8') as stop_in:

        tag_dict = {}
        tag_list = []
        # Prepare taglist
        stop_set = set(stop_in.read().split())
        tag_reader = csv.reader(tag_in)
        ind = 0
        for row in tag_reader:
            if row[1] not in stop_set:
                tag_dict[row[1]] = (ind, math.log10(float(row[2])))  # tag_dict[tag] = (ind, weight)
                tag_list.append((math.log10(float(row[2])), row[1]))  # tag_list[ind] = (weight, tag)
                ind += 1
        if flag_normalize_tag:
            tag_dict = normalize_tag_dict(tag_dict)
            tag_list = normalize_tag_list(tag_list)
        return tag_dict, tag_list


def prepare_cluster(language):
    cluster_centers = []

    # load cluster centers
    cur_center = db.cursor()
    cur_center.execute('USE ' + database_name + ';')

    # allocate memory
    cur_center.execute('SELECT count(*) FROM center;')
    # cur_center.execute('SELECT count(*) FROM center WHERE language="' + language + '";')
    n_center = cur_center.fetchall()[0][0]
    n_tag = len(tag_list)
    cluster_matrix = sp.sparse.lil_matrix((n_center, n_tag))

    # get center info
    cur_center.execute('SELECT * FROM center;')
    # cur_center.execute('SELECT * FROM center WHERE language="' + language + '";')
    i = 0
    for cluster_label, ind, repo_id, vec_pickled in cur_center:
    # for language, cluster_label, ind, repo_id, vec_pickled in cur_center:
        cluster_matrix[i, :] = pickle.loads(str(vec_pickled))
        center = {'ind': ind, 'id': repo_id}
        cluster_centers.append(center)
        i += 1

    return cluster_centers, cluster_matrix


tag_dict, tag_list = prepare_tag()
# cluster_centers, cluster_matrix = prepare_cluster()


def time_github_to_unix(timestr):
    return calendar.timegm(datetime.datetime.strptime(timestr, '%Y-%m-%dT%H:%M:%SZ').timetuple())


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


def get_repo_info_by_id(repo_id):
    url = "https://api.github.com/repositories/"
    full_url = url + str(repo_id)
    header = {'Authorization': token}
    req = urllib2.Request(full_url, headers=header)
    f = urllib2.urlopen(req)
    readin = f.read()
    f.close()
    datajson = json.loads(readin)
    return datajson


def get_repo_info(full_name):
    url = "https://api.github.com/repos/"
    full_url = url + full_name
    header = {'Authorization': token}
    req = urllib2.Request(full_url, headers=header)
    f = urllib2.urlopen(req)
    readin = f.read()
    f.close()
    datajson = json.loads(readin)
    return datajson


def get_readme(full_name):
    url = "https://api.github.com/repos/"
    folder = '/readme'
    full_url = url + full_name + folder
    header = {'Authorization': token}
    req = urllib2.Request(full_url, headers=header)
    f = urllib2.urlopen(req)
    readin = f.read()
    f.close()
    datajson = json.loads(readin)
    return base64.b64decode(datajson['content'])


def clean_markdown(text):
    rows = text.splitlines()
    result = []

    flag_skip = False
    for row in rows:
        row = row.decode('utf-8')
        if '```' in row:
            flag_skip = not flag_skip
            continue
        if flag_skip:
            continue

        while 'https://' in row:
            pos = row.find('https://')
            pos_end = row.find(' ', pos)
            row = " ".join([row[:pos], row[pos_end:]])

        while 'http://' in row:
            pos = row.find('http://')
            pos_end = row.find(' ', pos)
            row = " ".join([row[:pos], row[pos_end:]])

        result.append(row)

    return '\n'.join(result)


def text2vector(text):

    text = clean_markdown(text)

    fdist = FreqDist()
    for word in word_tokenize(text.lower()):
        if word in tag_dict:
            # fdist[word] += 1
            fdist[word] = 1

    # Convert to sparse vector
    vec = sp.sparse.lil_matrix((1, len(tag_dict)))  # row vec
    for item in fdist:
        ind = int(tag_dict[item][0])
        weight = tag_dict[item][1]
        vec[0, ind] = fdist[item] * weight

    # Normalize
    norm = np.sqrt(vec.dot(vec.transpose())[0, 0])
    if norm != 0:
        vec /= norm
    return vec


def normalize_vec(vec):
    # Normalize
    norm = np.sqrt(vec.dot(vec.transpose())[0, 0])
    if norm != 0:
        vec /= norm
    return vec


def weigh_vec(vec):
    iind, jind = vec.nonzero()
    for j in jind:
        vec[0, j] *= tag_list[j][0]

    norm = np.sqrt(vec.dot(vec.transpose())[0, 0])
    if norm != 0:
        vec /= norm
    return vec


def label_vec(vec, cluster_matrix):
    similarity_measures = cluster_matrix.dot(vec.transpose()).todense()
    return np.argmax(similarity_measures)


def load_similarity_matrix(label, language):
    cur_cluster = db.cursor()
    cur_cluster.execute('USE ' + database_name + ';')
    cur_cluster.execute('SELECT indlist, idlist, matrix FROM cluster WHERE cluster_label='+str(label) + ';')
    # cur_cluster.execute('SELECT indlist, idlist, matrix FROM cluster WHERE cluster_label='+str(label)+' and language="' + language + '";')

    indlist_text, idlist_text, matrix_pickled = cur_cluster.fetchall()[0]
    matrix = pickle.loads(str(matrix_pickled))
    indlist = json.loads(indlist_text)
    idlist = json.loads(idlist_text)

    return indlist, idlist, matrix


def get_full_name_weight_list(idlist):
    full_name_list = []
    weight_list = []
    cur = db.cursor()
    cur.execute('USE ' + database_name + ';')

    for repo_id in idlist:
        cur.execute('SELECT full_name, star, fork FROM readmelist WHERE id='+str(repo_id))
        row = cur.fetchall()[0]
        full_name_list.append(str(row[0]))
        n_star = int(row[1])
        n_fork = int(row[2])
        weight_list.append(math.log10(n_star + ratio_star_fork * n_fork))

    return full_name_list, weight_list


def generate_network(similarity_matrix, idlist, full_name_list, weight_list, thres=0.5):
    n_id = len(idlist)

    nodes = []
    for i in xrange(0, n_id):
        nodes.append({'id': idlist[i], 'value': weight_list[i]*node_scaling,
                      'label': full_name_list[i].split('/')[1]})  #, 'title': full_name_list[i]})

    # edges = []
    # weight_pair = []
    # for i in range(0, n_id):
    #     weight_pair.append((i, weight_list[i]))
    # sorted(weight_pair, key=lambda x: x[0], reverse=True)

    # print(similarity_matrix.shape)

    edge_info = []
    for i in xrange(0, n_id-1):
        for j in xrange(i+1, n_id):
            if similarity_matrix[i, j] >= thres:
                edge_info.append((i, j, similarity_matrix[i, j]))
    edge_info = sorted(edge_info, key=lambda x: x[2], reverse=True)

    if len(edge_info) > n_edge_limit:
        edge_info = edge_info[:n_edge_limit]

    # count occurance
    edge_occurance = [0]*n_id
    for i, j, val in edge_info:
        edge_occurance[i] += 1
        edge_occurance[j] += 1

    for (i, val) in enumerate(edge_occurance):
        if val == 0:
            similarity_matrix[i, i] = 0
            max_ind = 0
            max_val = 0
            for j in xrange(0, n_id):
                if max_val < similarity_matrix[i, j]:
                    max_val = similarity_matrix[i, j]
                    max_ind = j
            edge_info.append((i, max_ind, max_val))

    edges = []
    for item in edge_info:
        if item[2] >= 0:
            edges.append({'from': idlist[item[0]], 'to': idlist[item[1]], 'value': item[2]*edge_scaling})

    return {'nodes': nodes, 'edges': edges}


@app.route('/_query')
def query():
    print('Receive _query')
    print(datetime.datetime.now())
    
    full_name = request.args.get('repo', 0, type=str)
    repo_info = get_repo_info(full_name)
    readme = get_readme(full_name)
    vec_readme = text2vector(readme)
    vec_description = text2vector(repo_info['description'])
    nvec = normalize_vec(vec_readme+vec_description)
    language = repo_info['language']
    cluster_centers, cluster_matrix = prepare_cluster(language)


    label = label_vec(nvec, cluster_matrix)
    # print(label)

    indlist, idlist, matrix_tmp = load_similarity_matrix(label, language)
    similarity_matrix = matrix_tmp
    full_name_list, weight_list = get_full_name_weight_list(idlist)

    shape = matrix_tmp.shape

    if repo_info['id'] not in idlist:  # not exists in database
        similarity_matrix = np.zeros((shape[0]+1, shape[1]+1))
        similarity_matrix[0:-1, 0:-1] = matrix_tmp
        similarity_matrix[-1, -1] = 1
        n_idlist = len(idlist)

        # Prepare matrix
        mat = sp.sparse.lil_matrix((n_idlist, len(tag_list)))
        cur = db.cursor()
        cur.execute('USE ' + database_name + ';')
        for i in range(0, n_idlist):
            cur.execute('SELECT vec FROM readmefreqvec WHERE id='+str(idlist[i]))
            mat[i, :] = weigh_vec(pickle.loads(str(cur.fetchall()[0][0])))

        similarity_additional_vec = mat.dot(nvec.transpose()).todense()
        for i in range(0, n_idlist):
            similarity_matrix[i, -1] = similarity_additional_vec[i]
            similarity_matrix[-1, i] = similarity_additional_vec[i]
        idlist.append(repo_info['id'])
        full_name_list.append(str(repo_info['full_name']))
        n_star = int(repo_info['stargazers_count'])
        n_fork = int(repo_info['forks_count'])
        weight_list.append(math.log10(n_star + ratio_star_fork*n_fork))

    network_json = generate_network(similarity_matrix, idlist, full_name_list, weight_list, thres=thres_link)
    network_json['focus_id'] = repo_info['id']

    print('Return _query')
    print(datetime.datetime.now())
    
    return flask.json.dumps(network_json)


@app.route('/_repo_info')
def repo_info():
    repo_id = request.args.get('repo_id', 0, type=int)

    cur = db.cursor()
    cur.execute('USE ' + database_name + ';')
    cur.execute('SELECT full_name, description, language, created_at, pushed_at, '
                'star, fork, homepage, readme FROM readmelist WHERE id='+str(repo_id)+';')

    item = cur.fetchall()

    if len(item) != 0:
        full_name, description, language, created, pushed, star, fork, homepage, readme = item[0]
        cur.execute('SELECT vec FROM readmefreqvec WHERE id='+str(repo_id)+';')
        vec_pickled = cur.fetchall()[0][0]
        nvec = weigh_vec(pickle.loads(str(vec_pickled)))
    else:  # need to get info from github api
        datajson = get_repo_info_by_id(repo_id)
        datajson['readme'] = get_readme(datajson['full_name'])
        print(datajson)
        full_name = datajson['full_name']
        description = datajson['description']
        language = datajson['language']
        created = time_github_to_unix(datajson['created_at'])
        pushed = time_github_to_unix(datajson['pushed_at'])
        star = datajson['stargazers_count']
        fork = datajson['forks_count']
        homepage = datajson['homepage']
        readme = datajson['readme']
        nvec = text2vector(readme)

    iind, jind = nvec.nonzero()
    tags = []
    for j in jind:
        tags.append({'text': tag_list[j][1], 'weight': nvec[0, j]*tag_scale})


    full_name = full_name.split('/')
    user = full_name[0]
    repo = full_name[1]

    tags = sorted(tags, key=lambda x: x['weight'], reverse=True)
    created = time.strftime('%b %d, %Y', time.localtime(created))
    pushed = time.strftime('%b %d, %Y', time.localtime(pushed))

    data = {'user': user, 'repo': repo, 'description': description, 'language': language,
            'created': created, 'pushed': pushed,
            'star': int(star), 'fork': int(fork), 'homepage': homepage, 'tags': tags[:n_tag_on_display],
            'id': repo_id, 'readme': markdown.markdown(readme[:readme_len])}

    return flask.json.dumps(data)


@app.route("/jquery")
def index_jquery():
    return render_template('index_js.html') 


if __name__ == "__main__":                                                                                                                                                           app.run(debug=True, host='0.0.0.0', port=80)                                                                                                                                 
