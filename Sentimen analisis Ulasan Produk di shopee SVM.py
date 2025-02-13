#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import nltk

nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud, STOPWORDS
import re


# In[7]:


#memuat data
df = pd.read_csv('Documents/data/dataScraping_Daviena.csv', index_col=0)
#menampilkan data paling atas
df


# # Cleaning Data

# In[8]:


#hanya memilih kolom produk, ulasan, rating
df = df[['Produk', 'Ulasan', 'Rating']]


# In[9]:


df.shape


# In[10]:


df = df.drop_duplicates(subset='Ulasan')


# In[11]:


df = df.dropna()


# In[12]:


df.shape


# In[13]:


df


# # labeling

# In[14]:


label = []
for index, row in df.iterrows():
    if row["Rating"] == 5 or row["Rating"] == 4:
        label.append('positif')
    elif row["Rating"] == 3:
        label.append('netral')
    else:
        label.append('negatif')
        
df["label"] = label       


# In[15]:


df.head()


# In[16]:


#df.to_csv('Documents/data/labeling_indah.csv', index=False)
data_label = pd.read_csv('Documents/data/labeling_indah.csv', encoding='latin1')


# In[17]:


df.groupby('label').describe()


# # Preprocessing

# In[18]:


def clean_daviena_data(text):
  text = re.sub(r'@[A-Za-z0-9_]+', '', text)
  text = re.sub(r'#\w+', '', text)
  text = re.sub(r'RT[\s]+', '', text)
  text = re.sub(r'https?://\S+', '', text)
  text = re.sub(r'[^A-Za-z0-9 ]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()

  return text

df['Ulasan'] = df['Ulasan'].apply(clean_daviena_data)


# In[19]:


df['Ulasan'] = df['Ulasan'].str.lower()


# In[20]:


df.sample(5)


# In[21]:


#df.to_csv('Documents/data/casefolding_indah.csv', index=False)


# In[22]:


#reset nomor baris
df = df.reset_index(drop=True)


# In[23]:


df.shape


# In[24]:


def filter_tokens_by_length(dataframe, column, min_words, max_words):
    words_count = dataframe[column].astype(str).apply(lambda x: len(x.split()))
    mask = (words_count >= min_words) & (words_count <= max_words)
    filtered_df = dataframe[mask]
    return filtered_df

min_words = 2
max_words = 50
df = filter_tokens_by_length(df, 'Ulasan', min_words, max_words)


# In[25]:


#reset nomor baris data
df = df.reset_index(drop=True)


# In[26]:


df.head()


# In[27]:


Rating = df['Rating']


# In[28]:


df.shape


# In[29]:


#Normalisai
norm ={' muter ':' putar ', ' gk ':' tidak', ' profisional ' : ' profesional', ' skrng ' : ' sekarang ', ' uwang ':' uang ' ,'tiktokan ':'tiktok', ' yg ':' yang ', ' udh ':' udah ', 'wkwk ':' ', ' min ':' kak ', ' malem ':' malam', ' malem2 ':' malam ', ' sm ':' sama ', ' dy ':' dia ', ' lg ':' lagi ', ' skrg ':' sekarang ', ' ddpn ':' didepan ', ' makasi ':' makasih ', ' pertamaz ':' pertamax ', ' jg ':' juga ', ' donk ':' dong ', ' ikutann ':' ikutan ', ' banyakk ':' banyak ', ' twt ':' tweet', 'mantaap ':'mantap ', ' juarak':' juara ', 'daridulu ':'dari dulu ', 'siapp ':'siap ', ' gamau ':' tidak mau ', ' sll ':' selalu ', ' qu ':' aku ', ' krn ':' karena ', ' irii':' iri', ' muluu ':' terus ', 'mada ':'masa ', 'jgn ':'jangan ', ' jgn ':' jangan ', ' muluuu ':' terus ', 'ntar ':'nanti ', ' awtnya':' awetnya', 'gg ':'keren ', ' kerennn':' keren ', ' bisaa ':' bisa ', 'gaaa':'tidak ', " yg ": " yang ", ' nyampe':' sampai', ' nyampe ':' sampai ', ' lu ':' kamu ', ' ikhlaaasss ':' ikhlas ', ' gak ':' tidak ', ' klo ':' kalo ', ' amp ': ' sampai ', ' ga ':' tidak ', ' yaaaa':' ya ', 'betolll ':'betul ', ' kaga ':' tidak ', ' idk ':' tidak tahu ', ' jkt ':' jakarta ', ' lo ':' kamu ', ' bjir ':' ', ' kek ':' seperti ', ' yg ':' yang ', ' utk ':' untuk ', 'kismin ':'miskin ', ' kismin ':' miskin ', ' pd ':' pada ', ' dgn ':' dengan ', ' ituu ':' itu ', ' jg ':' juga ', 'yoi':'iya ', ' yoi ':' iya ', 'org2 ':'orang ', ' tak ':' tidak ', ' kyk ':' seperti ', ' sbg ':' sebagai ', ' anjjjj ':' ', ' bgt ':' banget ', 'km ':'kamu ', ' km ':' kamu', ' byk ':' banyak ', ' lg ':' lagi ', ' mrk ':' mereka ', ' blm ':' belum ',
        ' dgn ' : ' dengan ', ' seller ': ' penjual ',' service ':' pelayanan ', ' tp ':' tapi ', ' recommended ':' rekomendasi ', ' kren ':' keren ', ' kereen ':' keren ', ' mantab ': ' keren ',' matching ':' sesuai ','happy':' senang ','original': 'asli ','ori':'asli ', "trusted" : "terpercaya", "angjaaaassss":"keren", " gue ": " saya ", "bgmn ":" bagaimana ", ' tdk':' tidak ', ' blum ':' belum ', 'mantaaaaaaaappp':' bagus ', 'mantaaap':'bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', ' dg ':' dengan ', 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' recommended':' rekomen ', 'recomend':' rekomen ', 'good':' bagus ', " dgn " : " dengan ", " gue ": " saya ", " dgn ":" dengan ", "bgmn ":" bagaimana ", ' tdk':' tidak ', 
' blum ':' belum ', "quality":"kualitas", 'baguss':'bagus', 'overall' : 'akhirnya', 'mantaaaaaaaappp':' bagus ', ' josss ':' bagus ', ' thanks ': ' terima kasih ', 'fast':' cepat ', 
 'trims':' terima kasih ', 'brg':' barang ', 'gx':' tidak ', ' dgn ':' dengan ', ' real ': ' asli ', ' bnb ': ' baru ' ,
' recommended':' rekomen ', 'recomend':' rekomen ', 'good':'bagus',
'eksis ':'ada ', 'beenilai ':'bernilai ', ' dg ':' dengan ', ' ori ':' asli ', ' setting ':' atur ', " free ":" gratis ",
' yg ':' yang ', 't4 ':'tempat', ' awat ':' awet', ' mantep ':' bagus ', 'mantapp':'bagus', 
'kl ':'kalo', ' k ':' ke ', 'plg ':'pulang ', 'ajah ':'aja ', 'bgt':'banget', 'lbh ':'lebih', 'ayem':'tenang','dsana ':'disana ', 'lg':' lagi',
'pas ':'saat ', ' bnib ': ' baru ', 
' nggak ':' tidak ', 'karna ':'karena ', 'utk ':'untuk ',
' dn ':' dan ', ' mlht ':' melihat ', ' pd ':' pada ', 'mndngr ':'mendengar ', 'crita':'cerita', ' dpt ':' dapat ', ' mksh ':' terima kasih ', ' sellerrrr':' penjual', 'ori ':'asli ', ' new ':' baru ',
'sejrh':'sejarah', 'mnmbh ':'menambah ', 'sayapun':'saya', 'thn ':'tahun ', 'good':'bagus', ' awettt':' awet',
'halu ':'halusinasi ', ' nyantai ':' santai ', 'plus ':'dan ',
' ayang ':' sayang ', ' Rekomendded ':' direkomendasikan ', ' now ': ' sekarang ', 'slalu ':'selalu ', 'photo ': 'foto ', 'slah ':'salah ', 'krn':'karena', ' ga ':' tidak ', 'ok ':'oke ', ' meski':' mesti', ' para ':'parah', ' nawarin':' menawari', 'socmed':'sosial media',
' sya ':' saya ', 'siip':'bagus', ' bny ':' banyak ', ' tdk ':' tidak ', ' byk ':' banyak ', 
' pool ':' sekali ', " pgn ":" ingin ", " gue ":" saya ", " bgmn ":" bagaimana ", " ga ":" tidak ", 
' gak ':' tidak ', ' dr ':' dari ', ' yg ':' yang ', ' lu ':' kamu ', ' sya ':' saya ', 
' lancarrr ':' lancar ', ' kayak ':' seperti ', ' ngawur ':' sembarangan ', ' k ':' ke ', 
' luasss ':' luas ', ' sy ':' saya ', ' thn ':' tahun ', ' males ':' malas ',
' tgl ':' tanggal ', ' lg ':' lagi ',' bs ':' bisa ', ' bgtt ':' banget ',' gua ':' saya ', ' exp ':' ekspedisi ', 'exp ':'ekspedisi ', ' mantep ':' mantap ', ' bangettt ':' sangat ', ' ndak ':' tidak ', ' dluan ':' duluan ', ' packing ':' kemasan ', ' kliatan ':' kelihatan ', ' dgn ':' dengan'}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
        return str_text
    
    df['Ulasan'] = df['Ulasan'].apply(lambda x: normalisasi(x))


# In[30]:


df.head(5)


# In[31]:


get_ipython().system('pip install Sastrawi')


# In[32]:


#stopword
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
more_stop_words = []

stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)

new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def stopword(str_text):
  str_text = stop_words_remover_new.remove(str_text)
  return str_text

df['Ulasan'] = df['Ulasan'].apply(lambda x: stopword(x))


# In[33]:


df.sample(5)


# In[34]:


#df.to_csv('Documents/data/stopword_indah.csv', index=False)


# In[35]:


#tokenize
tokenized = df['Ulasan'].apply(lambda x:x.split())
tokenized


# In[96]:





# In[37]:


#stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stemming(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text_cleaning:
        dt = stemmer.stem(w)
        do.append(dt)
        d_clean = []
        d_clean = " ".join(do)
        print(d_clean)
        return d_clean
    
    tokenized = tokenized.apply(stemming)


# In[38]:


#tokenized.to_csv('Documents/data/tokenized_indah.csv', index=False)
data_clean = pd.read_csv('Documents/data/tokenized_indah.csv', encoding='latin1')
data_clean.sample(5)


# In[39]:


tokenized = pd.read_csv('Documents/data/tokenized_indah.csv', index_col=0)


# In[40]:


label


# In[41]:


at1 = pd.read_csv('Documents/data/tokenized_indah.csv')
at2 = pd.read_csv('Documents/data/labeling_indah.csv')
att2 = at2['label']

result = pd.concat([at1, att2], axis=1)


# In[42]:


result.head()


# # TF-IDF

# In[43]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# In[44]:


Ulasan = result['Ulasan']


# In[45]:


Ulasan.isnull().sum()


# In[46]:


Ulasan = Ulasan.fillna('tidak ada komentar')


# In[47]:


cv = CountVectorizer()
term_fit = cv.fit(Ulasan)

print(len(term_fit.vocabulary_))


# In[48]:


term_fit.vocabulary_


# In[49]:


#kolom pertama berarti jumlah dokumen
#kolom kedua berarti letak katanya
#kolom ketiga berarti hasil dari tf
term_frequency_all = term_fit.transform(Ulasan)
print(term_frequency_all)


# In[50]:


ulasan_tf = Ulasan[1]
print(ulasan_tf)


# In[51]:


term_freaquency = term_fit.transform([ulasan_tf])
print(term_freaquency )


# In[52]:


dokumen = term_fit.transform(Ulasan)
tfidf_transformer = TfidfTransformer().fit(dokumen)
print(tfidf_transformer.idf_)

tfidf = tfidf_transformer.transform(term_freaquency)
print(tfidf)


# # visualisasi (nlp)

# In[53]:


train_s0 = df [df ["Rating"] == 0]


# In[54]:


train_s0["Ulasan"] = train_s0["Ulasan"].fillna("tidak ada komentar")


# In[55]:


train_s0.head()


# In[56]:


from wordcloud import WordCloud


# In[57]:


train_s1 = df [df ["Rating"] == 1]


# In[58]:


train_s1["Ulasan"] = train_s1["Ulasan"].fillna(" Tidak Ada Komentar")


# In[59]:


train_s1.head()


# In[60]:


all_text_s1 = ' '.join(word for word in train_s1["Ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode="RGBA", background_color='white').generate(all_text_s1)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


# In[61]:


train_s2 = df [df ["Rating"] == 2]


# In[62]:


train_s2["Ulasan"] = train_s2["Ulasan"].fillna(" Tidak Ada Komentar")


# In[63]:


train_s2.head()


# In[64]:


all_text_s2 = ' '.join(word for word in train_s2["Ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode="RGBA", background_color='white').generate(all_text_s2)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


# In[65]:


train_s3 = df [df ["Rating"] == 3]


# In[66]:


train_s3["Ulasan"] = train_s3["Ulasan"].fillna(" Tidak Ada Komentar")


# In[67]:


train_s3.head()


# In[68]:


all_text_s3 = ' '.join(word for word in train_s3["Ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode="RGBA", background_color='white').generate(all_text_s3)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


# In[69]:


train_s4 = df [df ["Rating"] == 4]


# In[70]:


train_s4["Ulasan"] = train_s4["Ulasan"].fillna(" Tidak Ada Komentar")


# In[71]:


train_s4.head()


# In[72]:


all_text_s4 = ' '.join(word for word in train_s4["Ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode="RGBA", background_color='white').generate(all_text_s4)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


# In[73]:


train_s5 = df [df ["Rating"] == 5]


# In[74]:


train_s5["Ulasan"] = train_s5["Ulasan"].fillna(" Tidak Ada Komentar")


# In[75]:


train_s5.head()


# In[76]:


all_text_s5 = ' '.join(word for word in train_s5["Ulasan"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode="RGBA", background_color='white').generate(all_text_s5)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


# In[77]:


sentimen_data = pd.value_counts(df["Rating"], sort=True)
sentimen_data.plot(kind='bar', color=['lightskyblue', 'red'])
plt.title("Bar Chart")
plt.show


# # split data (TF-IDF)
# membagi presentase data. jika ada 100% di bagi presentasinya sentimen positif negatifnya

# In[78]:


result['Ulasan'] = result['Ulasan'].fillna("Tidak ada komentar")


# In[79]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(result['Ulasan'], result['label'],
                                                   test_size=0.1, stratify=result['label'], random_state=30)


# In[80]:


import numpy as np


# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')


# In[82]:


X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)


# In[83]:


X_train = X_train.toarray()


# In[84]:


X_test = X_test.toarray()


# # Machine Learning (SVM)

# In[85]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# In[86]:


svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# In[87]:


y_pred = svm_model.predict(X_test)


# In[88]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[89]:


print(svm_model.score(X_test, y_test))


# In[90]:


from sklearn.metrics import classification_report

pred = svm_model.predict(X_test)
print(classification_report(y_test, pred))


# In[91]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming y_true and y_pred are your actual and predicted labels

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print(classification_report(y_test, pred))


# In[92]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Membuat visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# # Evaluasi dan Grafik

# In[93]:


import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_score = accuracy_score(y_test, pred) 
pre_score = precision_score(y_test, pred, average='weighted')
rec_score = recall_score(y_test, pred, average='weighted')
f_score = f1_score(y_test, pred, average='weighted')

# Data evaluasi yang ingin diplot
scores = {
    'Accuracy': acc_score,
    'Precision': pre_score,
    'Recall': rec_score, 
    'F-Measure': f_score
}
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[94]:


eval_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Score']) 


# In[95]:


eval_df['Score'].plot(kind='line', marker='o', color='blue')
plt.title('Hasil Evaluasi Model')
plt.xlabel('Metrik') 
plt.ylabel('Nilai Rata-Rata')
plt.grid(True) 

plt.show()


# In[ ]:




