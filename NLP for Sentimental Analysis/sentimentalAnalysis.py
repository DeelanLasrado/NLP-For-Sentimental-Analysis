import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Wordcloud
from wordcloud import wordcloud


nltk.download('stopwords')
nltk.download('wordnet')

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from scikitplot.metrics import plot_confusion_matrix

'''#DATA
i didnt feel humiliated;sadness
i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake;sadness
im grabbing a minute to post i feel greedy wrong;anger
i am ever feeling nostalgic about the fireplace i will know that it is still on the property;love
i am feeling grouchy;anger
'''


df_train = pd.read_csv("C:\\Users\\deela\\Downloads\\archive (3)\\train.txt",delimiter=';', names=['text','label'])
df_val = pd.read_csv("C:\\Users\\deela\\Downloads\\archive (3)\\val.txt", delimiter=';', names=['text','label'])

print(df_train)
'''                                                text    label
0                                i didnt feel humiliated  sadness
1      i can go from feeling so hopeless to so damned...  sadness
2       im grabbing a minute to post i feel greedy wrong    anger
3      i am ever feeling nostalgic about the fireplac...     love'''



#to combine both training and validation data
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

print(df.label.value_counts())


# Positive Sentiments - Joy, Love, Surprise - 1
# Negative Sentiments - Anger, Sadness, Fear - 0

df['label'].replace(to_replace=['surprise', 'joy', 'love'], value=1, inplace=True)
df['label'].replace(to_replace=['anger', 'sadness','fear'], value=0, inplace=True)


print(df.label.value_counts())


#Lemmatization - to reduce the word to base form  ex: went -> go
lm = WordNetLemmatizer()



def tranformation(df_column):
  output = []
  for i in df_column:
    new_text = re.sub('[^a-zA-Z]',' ',str(i))# Replace all non-alphanumeric characters in the string with spaces
    new_text = new_text.lower()
    new_text = new_text.split()
    new_text = [lm.lemmatize(j) for j in new_text if j not in set(stopwords.words('english'))]
    output.append(' '.join(str(k) for k in new_text))
  
  return output


var = tranformation(df.text)

#print(var)
'''['didnt feel humiliated', 'go feeling hopeless damned hopeful around someone care awake', 
'im grabbing minute post feel greedy wrong', 'ever feeling nostalgic fireplace know still property', 'feeling grouchy',...] '''

'''
# Word Cloud
from wordcloud import WordCloud
plt.figure(figsize=(50,28))
word = ''
for i in var:
  for j in i:
    word += " ".join(j)
print(word) #word= everthing is converted to single sentence

#didnt feel humiliatedgo feeling hopeless damned hopeful around someone care awakeim grabbing minute post 
    feel greedy wrongever feeling nostalgic fireplace know still propertyfeeling grouchyive feeling



wc = WordCloud(width=1000, height= 500, background_color='white', min_font_size=10).generate(word)
plt.imshow(wc)
plt.show()'''


# Bag of Words model (BOW)
cv = CountVectorizer(ngram_range=(1,2))
'''The CountVectorizer object is used to create a bag of words model (BOW) from a text corpus.
        The BOW model is a representation of the text corpus where each document is represented as a vector of word counts.'''
#bow model
'''| Word | Count |
|---|---|---|
| politics | 2 |
| government | 1 |
| bill | 1 |
| passed | 1 |'''

traindata = cv.fit_transform(var)
X_train = traindata
y_train = df.label


#model selection
model = RandomForestClassifier()

'''# Hyper Parameter Tuning

parameters = {'max_features':('auto', 'sqrt'),
              'n_estimators': [500, 1000, 1500],
              'max_depth': [5,10, None],
              'min_samples_leaf':[1, 2, 5, 10],
              'min_samples_split':[5, 10, 15],
              'bootstrap':[True, False]}

grid_search = GridSearchCV(model, 
                           parameters, 
                           cv=5,
                           return_train_score = True,
                           n_jobs=1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

rfc = RandomForestClassifier(max_features= grid_search.best_params_['max_features'],
                             n_estimators= grid_search.best_params_['n_estimators'],
                             max_depth= grid_search.best_params_['max_depth'],
                             min_samples_leaf= grid_search.best_params_['min_samples_leaf'],
                             min_samples_split= grid_search.best_params_['min_samples_split'],
                             bootstrap= grid_search.best_params_['bootstrap'])

rfc.fit(X_train, y_train)



y_pred =rfc.predict('i didnt feel humiliated')'''

model.fit(X_train,y_train)
'''y_pred=model.predict('i didnt feel humiliated')
print(y_pred)'''



def sentimental_analysis(input):
  new_input = tranformation(input)
  transformed_input = cv.transform(new_input)
  prediction = model.predict(transformed_input)
  print(prediction)
  if prediction[0] == 0:
    print('Negative Sentiment')
  elif prediction == 1:
    print('Positive Sentiment')
  else:
    print('Invalid Sentiment')


input = "Today I was playing in the park and I fell"
sentimental_analysis(input)