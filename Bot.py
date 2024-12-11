# Basic Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
import re
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
# Classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
# Performance Matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


import pickle
import os
import warnings
warnings.filterwarnings("ignore")

#read data file
df = pd.read_csv('data/Chatbot Dataset.csv',encoding='ISO-8859-1')
df = df.dropna(axis=0)
#remove dupicates
df_unduplicate = df.drop_duplicates(keep=False,inplace=False)
print(df_unduplicate)

#show data shape and columns
print('Dataset size:',df.shape)
print('Columns are:',df.columns)
Y = df['Intent']

df.info()

counter=collections.Counter(df['Intent'])
print(counter)

#xchar are special characters in text such as emojis for uniformity
xchar = pd.read_csv('data/Xchar.txt',sep=',',header=None)
xchar_dict = {i:j for i,j in zip(xchar[0],xchar[1])}
pattern = '|'.join(sorted(re.escape(k) for k in xchar_dict))

def replace_xchar(text):
    text = re.sub(pattern,lambda m: xchar_dict.get(m.group(0)), text, flags=re.IGNORECASE)
    return text

def remove_punct(text):
    text = replace_xchar(text)
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

#import contraction dictionary
#removing and reconstracting all the shortened words in file and replacing them with their full writen form allowing sentiment analysis
import json
with open('data/contractionDic.json') as f:
    cont_data = f.read()
js = json.loads(cont_data)
print("Data type after reconstruction : ", type(js))
print(js)

contraction_dict = json.loads(cont_data)
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

#breaking apart original text into individual pieces for further analysis
def tokenization(text):
    text = text.lower()
    text = re.split('\W+', text)
    return text

nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')
stopword.extend(['patients', 'may','day', 'case','old','u','n','didnt','yr', 'year', 'woman', 'man', 'girl','boy',
                 'brother','dad','one', 'two', 'sixteen', 'yearold', 'fu', 'weeks', 'week',
              'treatment', 'associated','ive','ate','feel','keep'
                'basic','im'])
#remove all stop words
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text

nltk.download('wordnet')

#lemmatization is used to normalize text and prepare words and documents for further processing in Machine Learning
Wlament = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [Wlament.lemmatize(word) for word in text]
    return text
#text cleaning
def clean_text(text):
    text = replace_contractions(text)
    text = remove_punct(text)
    text = tokenization(text)
    # text = remove_stopwords(text)
    text = lemmatizer(text)
    return text

#count vectorizer
#Creating a matrix for each word with a unique column matrix represented in a table where the value of each cell is the count of each word

sample =['the dog that flies landed', 'cows drink milk',  'out of the fring pan into the fire'] #upsard sentenses

# Create a Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(sample)
# Encode the Document
vector = vectorizer.transform(sample)

print("Encoded sample is:")
print(vector.toarray())

sample_out = sorted(vectorizer.vocabulary_)

print(sample_out)

sns.heatmap(vector.toarray(), annot=True, cbar=False, xticklabels=sample_out,
                                             yticklabels=['Sentence 1','Sentence 2','Sentence 3'])

#Term frequency using inverse document frequency
#the measure of how requently a term appears

from sklearn.feature_extraction.text import TfidfVectorizer
sample =['cows drink all the milk', 'the talking cat went home', 'out of the fring pan into the fire'] #upsard sentenses

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(sample).toarray()

print (tfidf)

print (tfidf_vectorizer.vocabulary_)

sorted_sample = sorted(tfidf_vectorizer.vocabulary_)

print(sorted_sample)
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=sorted_sample,
                                           yticklabels=['Sentence 1','Sentence 2','Sentence 3'])

#split training data into two and Apply feature extraction
X_train, X_test, y_train, y_test = train_test_split(df['User'], df['Intent'],test_size=0.25, random_state = 31)

countVectorizer1 = CountVectorizer(analyzer=clean_text)
countVector1 = countVectorizer1.fit_transform(X_train)

countVector2 = countVectorizer1.transform(X_test)

tfidf_transformer_xtrain = TfidfTransformer()
x_train = tfidf_transformer_xtrain.fit_transform(countVector1)

tfidf_transformer_xtest = TfidfTransformer()
x_test = tfidf_transformer_xtest.fit_transform(countVector2)

#using svm (support vector machine) and mlp (multi layer perceptron) to perform accuracy, recall, precision and F1-score
#svm

svm = SGDClassifier()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

svm_acc = accuracy_score(y_pred, y_test)
svm_prec = precision_score(y_test, y_pred, average='macro')
svm_recal = recall_score(y_test, y_pred, average='macro')
svm_cm = confusion_matrix(y_test,y_pred)
svm_f1 = f1_score(y_test, y_pred, average='macro')

print('Accuracy:', '{0:.3f}'.format(svm_acc*100))
print('Precision:', '{0:.3f}'.format(svm_prec*100))
print('Recall:', '{0:.3f}'.format(svm_recal*100))
print('F1-score:', '{0:.3f}'.format(svm_f1*100))
print(classification_report(y_test,y_pred))

#mlp
mlp = MLPClassifier(random_state=6, max_iter=97)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)

mlp_acc = accuracy_score(y_pred, y_test)
mlp_prec = precision_score(y_test, y_pred, average='macro')
mlp_recal = recall_score(y_test, y_pred, average='macro')
mlp_cm = confusion_matrix(y_test,y_pred)
mlp_f1 = f1_score(y_test, y_pred, average='macro')

print('Accuracy:', '{0:.3f}'.format(mlp_acc*100))
print('Precision:', '{0:.3f}'.format(mlp_prec*100))
print('Recall:', '{0:.3f}'.format(mlp_recal*100))
print('F1-score:', '{0:.3f}'.format(mlp_f1*100))
print(classification_report(y_test,y_pred))

#function to determin the percenteage of each emotional call made by the bot when interacting
def get_percentage_emotinal_call(cm):
    per_emotion_precision = []
    for i in range(len(cm)):
        count_per_emotion,accurate = 0,0
        for j in range(len(cm)):
            if i == j:
                accurate = cm[j][i]
            count_per_emotion += cm[j][i]
        per_emotion_precision.append(round((accurate/count_per_emotion)*100,3))

    per_emotion_recall = []
    for i in range(len(cm)):
        count_per_emotion,accurate = 0,0
        for j in range(len(cm)):
            if i == j:
                accurate = cm[i][j]
            count_per_emotion += cm[i][j]
        per_emotion_recall.append(round((accurate/count_per_emotion)*100,3))

    return per_emotion_precision, per_emotion_recall

#get precision of each model
svm_per_prec,svm_per_recall = get_percentage_emotinal_call(svm_cm)
mlp_per_prec,mlp_per_recall = get_percentage_emotinal_call(mlp_cm)
#recal the models used
per_precision_list = pd.DataFrame({'Support vector Machine' : svm_per_prec, 'Multi Layer Perceptron' : mlp_per_prec},
                                index=svm.classes_)
per_precision_list

print(len(set(y_test)))
import collections
counter=collections.Counter(y_test)
print(counter)

#save the models to disk
with open('models/Chatbot_Model_final.pkl','wb') as f:
    pickle.dump([svm, mlp], f)

#load models from disc
if os.path.isfile('models/Chatbot_Model_final.pkl'):
    # Getting back the objects:
    with open('models/Chatbot_Model_final.pkl','rb') as f:
        svm, mlp = pickle.load(f)
        print('File was Loaded Successfully')
else:
    print('File can not be Found')

import random
def response_generator(text, intent_name):
    reply = response(text, intent_name)

    return reply

def recall_countvectorizer(s1, s2):

    # turn input sentences to list from
    allsentences = [s1 , s2]

    from scipy.spatial import distance

    # turn all text into vectors
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)

    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    # find distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    return round((1-cosine),2)

def response(text, intent_name):
    maximum = float('-inf')
    response = ""
    closest = ""
    replies = {}
    list_sim, list_replies = [],[]
    dataset = df[df['Intent']==intent_name]
    for i in dataset.iterrows():
        sim = recall_countvectorizer(text, i[1]['User'])
        list_sim.append(sim)
        list_replies.append(i[1]['Chatbot'])

    for i in range(len(list_sim)):
        if list_sim[i] in replies:
            replies[list_sim[i]].append(list_replies[i])
        else:
            replies[list_sim[i]] = list()
            replies[list_sim[i]].append(list_replies[i])
    d1 = sorted(replies.items(), key = lambda pair:pair[0],reverse=True)
    return d1[0][1][random.randint(0,len(d1[0][1])-1)]

# using ensamble learning to train the models
accuracies = np.array([svm_acc, mlp_acc])
norm_accuracy = accuracies - min(accuracies)
model_weight = norm_accuracy / sum(norm_accuracy)  #SVM, MLP
Intents = df['Intent'].unique()
def extract_best_intent(list_intent_pred):
    intent_scores = {}
    for intent in Intents:
        intent_scores[intent] = 0.0
    for i in range(len(list_intent_pred)):
        intent_scores[list_intent_pred[i]] += model_weight[i]
    si = sorted(intent_scores.items(), key = lambda pair:pair[1],reverse=True)[:7]
    return si[0][0],si

# adding weights to train the model
accuracies = np.array([svm_acc, mlp_acc])
norm_accuracy = accuracies - min(accuracies)
model_weight = norm_accuracy / sum(norm_accuracy)
print(model_weight)
print('ready to server ')


from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np
import logging
import pickle
from typing import Dict, List, Set
import json

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)


# Struktur data untuk menyimpan informasi
class ChatDatabase:
    def __init__(self):
        self.active_users: Dict[str, Dict] = {}
        self.messages: List[Dict] = []
        self.group_chat_users: Set[str] = set()
        self.emotion_history: Dict[str, List] = {}  # Menyimpan history emosi per user


database = ChatDatabase()

# ====================================
# FUNGSI UTILITAS
# ====================================

def log_activity(activity: str, user: str = "System") -> None:
    """
    Mencatat aktivitas server

    Parameters:
        activity (str): Aktivitas yang dicatat
        user (str): Pengguna yang melakukan aktivitas
    """
    logging.info(f"{user}: {activity}")

def validate_request_data(data: Dict, required_fields: List[str]) -> bool:
    """
    Memvalidasi data request

    Parameters:
        data (Dict): Data yang akan divalidasi
        required_fields (List[str]): Daftar field yang wajib ada

    Returns:
        bool: Status validasi
    """
    return all(field in data for field in required_fields)



# def response_generator(user_input, best_intent):
#     # Placeholder for response generation logic
#     return f"I see you're interested in {best_intent}. How can I assist you further?"

def extract_best_intent(list_intent):
    # Placeholder for intent extraction logic
    return max(set(list_intent), key=list_intent.count), None


# ====================================
# ROUTE API
# ====================================

@app.route('/api/join', methods=['POST'])
def join_chat():
    """
    Handler untuk bergabung ke chat
    """
    data = request.json

    if not validate_request_data(data, ['username']):
        return jsonify({
            'status': 'error',
            'message': 'Username diperlukan'
        }), 400

    username = data['username']
    chat_type = data.get('chat_type', 'Group')

    # Update data pengguna
    database.active_users[username] = {
        'last_seen': datetime.now(),
        'chat_type': chat_type,
        'emotion_state': 'netral'  # Status emosi awal
    }

    if chat_type == 'Group':
        database.group_chat_users.add(username)
        join_message = f"SISTEM: {username} bergabung ke dalam grup chat"
        database.messages.append({
            'message': join_message,
            'timestamp': datetime.now().isoformat(),
            'username': 'SYSTEM',
            'chat_type': 'Group',
            'emotion': 'system'
        })

        log_activity(f"{username} bergabung ke grup chat")

    return jsonify({
        'status': 'success',
        'message': f'Selamat datang {username}!',
        'active_users': list(database.group_chat_users)
    })

@app.route('/api/leave', methods=['POST'])
def leave_chat():
    """
    Handler untuk keluar dari chat
    """
    data = request.json

    if not validate_request_data(data, ['username']):
        return jsonify({
            'status': 'error',
            'message': 'Username diperlukan'
        }), 400

    username = data['username']
    chat_type = data.get('chat_type', 'Group')

    if username in database.active_users:
        del database.active_users[username]

        if chat_type == 'Group':
            database.group_chat_users.remove(username)
            leave_message = f"SISTEM: {username} meninggalkan grup chat"
            database.messages.append({
                'message': leave_message,
                'timestamp': datetime.now().isoformat(),
                'chat_type': 'Group',
                'emotion': 'system'
            })

            log_activity(f"{username} keluar dari grup chat")

    return jsonify({
        'status': 'success',
        'message': f'Sampai jumpa {username}!'
    })

@app.route('/api/message', methods=['POST'])
def chat():
    """
    Handler untuk menerima pesan
    """
    data = request.json

    if not validate_request_data(data, ['message']):
        return jsonify({
            'status': 'error',
            'message': 'Pesan diperlukan'
        }), 400
    else:
        print("its validate")


    username = data['username']
    message = data['message']
    chat_type = data.get('chat_type', 'Group')
    emotion = data.get('emotion', 'netral')

    print(message)
    print(chat_type)
    print(emotion)
    print(username)
    # user_input = request.json.get('message')
    # user_name = request.json.get('name', 'User')

    # Process input
    processed_text = tfidf_transformer_xtest.fit_transform(countVectorizer1.transform([message]))

    print(processed_text)
    # Predict intents using SVM and MLP
    svm_intent = svm.predict(processed_text)[0]
    mlp_intent = mlp.predict(processed_text)[0]

    # Generate response
    list_intent = [svm_intent, mlp_intent]
    best_intent, _ = extract_best_intent(list_intent)

    bot_response = response_generator(message, best_intent)

     # Simpan pesan dengan informasi tambahan
    message_data = {
        'username':'Bot',
        'message': bot_response,
        'timestamp': datetime.now().isoformat(),
        'chat_type': chat_type,
        'emotion': best_intent
    }

   
    database.messages.append(message_data)


    print(f"ini di database : {message_data}")
    print(f"ini di response bot : {bot_response}")
    log_activity(f"Pesan baru diterima: {message[:50]}...")

    return jsonify({
        'status': 'success',
        'message': bot_response,
        'username': 'Bot'
    })

@app.route('/api/messages', methods=['GET'])
def get_messages():
    """
    Handler untuk mengambil pesan
    """
    chat_type = request.args.get('chat_type', 'Group')
    username = request.args.get('username', 'Bot')

    # Filter pesan sesuai tipe chat
    if chat_type == 'Group':
        filtered_messages = [
            msg for msg in database.messages
            if msg['chat_type'] == 'Group'
        ]
    else:
        filtered_messages = [
            msg for msg in database.messages
            if msg['chat_type'].startswith('Personal')
               and username in msg['chat_type']
        ]

    return jsonify({
        'messages': filtered_messages,
        'active_users': list(database.group_chat_users)
    })

@app.route('/api/emotions/<username>', methods=['GET'])
def get_user_emotions(username):
    """
    Handler untuk mengambil history emosi pengguna
    """
    if username in database.emotion_history:
        return jsonify({
            'status': 'success',
            'emotions': database.emotion_history[username]
        })
    return jsonify({
        'status': 'error',
        'message': 'Riwayat emosi tidak ditemukan'
    }), 404


@app.route('/api/active_users', methods=['GET'])
def get_active_users():
    """
    Handler untuk mendapatkan daftar pengguna aktif
    """
    return jsonify({
        'active_users': list(database.group_chat_users),
        'total_users': len(database.active_users)
    })


@app.route('/api/user/status', methods=['POST'])
def update_user_status():
    """
    Handler untuk memperbarui status pengguna
    """
    data = request.json
    if not validate_request_data(data, ['username', 'status']):
        return jsonify({
            'status': 'error',
            'message': 'Username dan status diperlukan'
        }), 400

    username = data['username']
    status = data['status']

    if username in database.active_users:
        database.active_users[username]['status'] = status
        return jsonify({
            'status': 'success',
            'message': f'Status {username} diperbarui'
        })
    return jsonify({
        'status': 'error',
        'message': 'Pengguna tidak ditemukan'
    }), 404


# ====================================
# KONFIGURASI SERVER
# ====================================

def setup_server():
    """
    Melakukan setup awal server
    """
    log_activity("Server bot dimulai")
    # Tambahkan konfigurasi tambahan di sini jika diperlukan


if __name__ == '__main__':
    setup_server()
    # Jalankan server dengan konfigurasi yang aman
    app.run(
        debug=True,  # Matikan saat production
        host='0.0.0.0',
        port=5002,
        threaded=True  # Mendukung multiple connections
    )