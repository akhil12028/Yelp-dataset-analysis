import pandas as pd
import json


with open('C:/Users/akhil/Desktop/yelp_dataset_challenge_round9/yelp_academic_dataset_checkin.json',encoding='utf8') as f:
    data1 = f.readlines()
checkin=[]
for i in range(0,len(data1)):
	checkin.append(json.loads(data1[i]))
	

with open('C:/Users/akhil/Desktop/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json',encoding='utf8') as f:
    data2 = f.readlines()
business=[]
for i in range(0,len(data2)):
	business.append(json.loads(data2[i]))
	
with open('C:/Users/akhil/Desktop/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json',encoding='utf8') as f:
    data3 = f.readlines()
reviews=[]
for i in range(0,len(data3)):
	reviews.append(json.loads(data3[i]))

city_restaurants=[]
x="Restaurants"
for b in business:
    if(b['city']=="Pittsburgh" or b['city']=="Charlotte"):
        if b['categories']!=None:
            if x in b['categories']:
                city_restaurants.append(b['business_id'])
                
checkin_restaurants=[]    
for c in checkin:
    if c['business_id'] in city_restaurants:
        checkin_restaurants.append([c['business_id'],len(c['time'])])
        
df_checkin = pd.DataFrame(checkin_restaurants)
df_checkin1= df_checkin.sort_values(1,axis=0,ascending=False)

count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
for i in range(len(df_checkin1)):
    if df_checkin1.iloc[i][1]<=20:
        count1=count1+1
    elif df_checkin1.iloc[i][1]<=40:
        count2=count2+1
    elif df_checkin1.iloc[i][1]<=60:
        count3=count3+1
    elif df_checkin1.iloc[i][1]<=80:
        count4=count4+1
    elif df_checkin1.iloc[i][1]<=100:
        count5=count5+1
    elif df_checkin1.iloc[i][1]<=120:
        count6=count6+1
    elif df_checkin1.iloc[i][1]<=140:
        count7=count7+1
    else:  
        count8=count8+1

print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
print(count6)
print(count7)
print(count8)
        

#Pittsburgh

city_restaurants=[]
x="Restaurants"
for b in business:
    if(b['city']=="Pittsburgh"):
        if b['categories']!=None:
            if x in b['categories']:
                city_restaurants.append(b['business_id'])
                
checkin_restaurants=[]    
for c in checkin:
    if c['business_id'] in city_restaurants:
        checkin_restaurants.append([c['business_id'],len(c['time'])])
        
df_checkin = pd.DataFrame(checkin_restaurants)
df_checkin = df_checkin.sort_values(1,axis=0,ascending=False).head(100)

popular_checkin=[]
for i in range(len(df_checkin)):
    popular_checkin.append(df_checkin.iloc[i][0])
    

review_restaurants=[]
for r in reviews:
    if r['business_id'] in popular_checkin:
        review_restaurants.append([r['business_id'],r['stars']])

df_reviews = pd.DataFrame(review_restaurants)
df_reviews = df_reviews.groupby(df_reviews[0])[1].mean()        
df_reviews = df_reviews.sort_values(0,ascending=False).head(50)

popular_restaurants1=[]
for i in range(len(df_reviews)):
    popular_restaurants1.append(df_reviews.index[i])

restaurant_type=[]
for b in business:
    if b['business_id'] in popular_restaurants1:
        if b['categories']!=None:    
            for type in b['categories']:
                if type != x:
                    restaurant_type.append(type)
        
type_count=[]
for x in set(restaurant_type):
    count = restaurant_type.count(x)
    type_count.append([count,x])
    
type_count.sort(reverse=True)
print("Pittsburgh")
print(type_count)

#Charlotte

city_restaurants=[]
x="Restaurants"
for b in business:
    if(b['city']=="Charlotte"):
        if b['categories']!=None:
            if x in b['categories']:
                city_restaurants.append(b['business_id'])
                
checkin_restaurants=[]    
for c in checkin:
    if c['business_id'] in city_restaurants:
        checkin_restaurants.append([c['business_id'],len(c['time'])])
        
df_checkin = pd.DataFrame(checkin_restaurants)
df_checkin = df_checkin.sort_values(1,axis=0,ascending=False).head(100)

popular_checkin=[]
for i in range(len(df_checkin)):
    popular_checkin.append(df_checkin.iloc[i][0])
    

review_restaurants=[]
for r in reviews:
    if r['business_id'] in popular_checkin:
        review_restaurants.append([r['business_id'],r['stars']])

df_reviews = pd.DataFrame(review_restaurants)
df_reviews = df_reviews.groupby(df_reviews[0])[1].mean()        
df_reviews = df_reviews.sort_values(0,ascending=False).head(50)

popular_restaurants2=[]
for i in range(len(df_reviews)):
    popular_restaurants2.append(df_reviews.index[i])

restaurant_type=[]
for b in business:
    if b['business_id'] in popular_restaurants2:
        if b['categories']!=None:    
            for type in b['categories']:
                if type != x:
                    restaurant_type.append(type)
        
type_count=[]
for x in set(restaurant_type):
    count = restaurant_type.count(x)
    type_count.append([count,x])
    
type_count.sort(reverse=True)
print("Charlotte")
print(type_count)



visualization=[]
for b in business:
    if b['business_id'] in popular_restaurants1:
       visualization.append([b['latitude'],b['longitude'],b['city'],b['postal_code']]) 

df_visualization=pd.DataFrame(visualization)
df_visualization.to_excel('visualization1.xlsx')

visualization=[]
for b in business:
    if b['business_id'] in popular_restaurants2:
       visualization.append([b['latitude'],b['longitude'],b['city'],b['postal_code']]) 

df_visualization=pd.DataFrame(visualization)
df_visualization.to_excel('visualization2.xlsx')

chinese_restaurants=[]
for b in business:
    if b['categories']!=None:
        if "Chinese" in b['categories'] and "Restaurants" in b['categories']:
            chinese_restaurants.append(b['business_id'])

chinese_reviews=[]           
for r in reviews:
    if r['business_id'] in chinese_restaurants:
        chinese_reviews.append(r['text'])
        
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()        
texts = []

for i in chinese_reviews:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stemmed_tokens)

words=[]
for t in texts:
    for p in t:
        words.append(p)

words_count=[]    
x=Counter(words)
words_count=[(l,k) for k,l in sorted([(j,i) for i,j in x.items()], reverse=True)]         
df_words=pd.DataFrame(words_count)
print(df_words.head(20))

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
p=ldamodel.print_topics(num_topics=5, num_words=5)
for i in p:
    print (i[1])