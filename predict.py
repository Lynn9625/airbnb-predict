import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import seaborn as sns
import urllib
from math import sin, cos, sqrt, atan2, radians
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate



# Import NYC Airbnb data
house = pd.read_excel("NYC_Airbnb.xlsx")
house.head()

# Plot for NYC Airbnb Room Type By Borough
sns.catplot(y="Borough", hue="Room_type", kind="count",
            palette="pastel", edgecolor=".6",
            data=house)

# Loading the png NYC image found on Google 
p=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_map=plt.imread(p)
plt.figure(figsize=(15,12))
plt.imshow(nyc_map,zorder=0,extent=[-74.258, -73.7, 40.48,40.92])
ax=plt.gca()
# Scatterplot for NYC Airbnb Price
house.plot(kind='scatter', x='longitude', y='latitude', label='NYC Airbnb Price',
           c='price', ax=ax, cmap=plt.get_cmap('gist_stern'), colorbar=True, alpha=0.4, zorder=3)
plt.legend()
plt.show()

# Remove outliers in price above 99% quantile ($750.0)
house['price']=house['price'].astype('str')
house['price']=house['price'].str.replace('$','')
house['price']=house['price'].str.replace(',','')
house['price']=house['price'].astype('float')
house.price.quantile(0.99)

# Divide price into 3 categories (low, medium and high) equally
house = house.copy()
house=house[house['price']<=750]
house['price']=pd.qcut(house['price'],3,labels=["low", "medium", "high"])

# Distribution of price in five boroughs of NYC
sns.catplot(y="Borough", hue="price", kind="count",
            palette="Blues", edgecolor=".6",
            data=house)

# Separate the amenities and list the 20 most frequently mentioned amenities
amenities_re=','.join(house.amenities)
amenities_re=amenities_re.replace('{', '')
amenities_re=amenities_re.replace('}', '')
amenities_re=amenities_re.replace('"', '')
counts=pd.Series(amenities_re.split(',')).str.lower().value_counts()
amenity = list(counts.index[:20])

# Create dummy variables for the 20 most frequently mentioned amenities. 1 indicates this observation has this ammenity and 0 indicates it doesn't
for i in amenity:
    index = house['amenities'].str.lower().str.contains(i)
    house[i] = np.where(index,1,0)
house.head()

# Calculate distance to the closest subway station
subway = pd.read_csv("SUBWAY_STATION.csv")
subway.head()
subway["LONGITUDE"] = subway["the_geom"].str.split(' ',expand=True)[1].str[1:-1]
subway["LATITUDE"] = subway["the_geom"].str.split(' ',expand=True)[2].str[0:-1]
distances = []
house_lo = list(house['longitude'])
subway_lo = list(map(float, subway['LONGITUDE']))
house_la = list(house['latitude'])
subway_la = list(map(float, subway['LATITUDE']))
for i in range(len(house_lo)):
    dis = []
    for j in range(len(subway_lo)):
        delta_x = radians(house_lo[i]) - radians(subway_lo[j])
        delta_y = radians(house_la[i]) - radians(subway_la[j])

        a = sin(delta_y / 2)**2 + cos(radians(subway_la[j])) * cos(radians(house_la[1])) * sin(delta_x / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = 6371 * c
        dis.append(d)
    min_dis = min(dis)
    distances.append(min_dis)
house = house.copy()
house['distance'] = distances

# Calculate the days from the hosting starts to the scraped date
house = house.copy()
house['host_since'] = house['host_since'].astype('datetime64[ns]') 
house['last_scraped'] = house['last_scraped'].astype('datetime64[ns]') 
house['days'] = house['last_scraped'] - house['host_since']
house['days'] = house['days'].astype(str)
house['days'] = house['days'].str.split(' ',expand=True)[0]

# Drop the missing values in hosting days
house = house.copy()
index = house[house['days'] == "NaT"].index
house.drop(index, inplace = True)
house['days'] = pd.to_numeric(house['days'])

# Drop unnecessary columns
house = house.drop(columns = ['id', 'last_scraped', 'host_id','space', 'house_rules', 'host_name', 'host_since', 'host_about', 'amenities', 'zipcode', 'latitude', 'longitude', 'neighbourhood'])

# Build models
x = house.drop('price',axis=1)
y = house['price']

x_dum=pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x_dum, y, test_size=0.3)

DTree=DecisionTreeClassifier()
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

gbm = GradientBoostingClassifier(random_state=0)
gbm.fit(x_train,y_train)
y_predict = gbm.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

n = [20,50,100,150,200]
depth = [2, 4, 6, None]
for i in n:
    for j in depth:
        clf =  RandomForestClassifier(random_state=0,n_estimators= i, max_depth = j)
        cv = cross_validate(clf, x_train, y_train, cv=5)
        print(i, j, cv['test_score'].mean())
rfc = RandomForestClassifier(random_state=0,n_estimators=150)
rfc.fit(x_train, y_train)
rfc_predict = rfc.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, rfc_predict))


# Create a dateframe listing the importance of the predictors from the most to the least
importance = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
col = list(x_dum.columns)
indices = np.argsort(importance)[::-1]
rank = [col[i] for i in indices]
pred = []
imp = []

for f in range(x_dum.shape[1]):
    pred.append(rank[f])
    imp.append(importance[indices[f]])

df = pd.DataFrame({
    'Predictors': pred,
    'Importance': imp
})

print(df)