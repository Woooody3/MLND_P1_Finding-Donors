import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

data = pd.read_csv("census.csv")
print data.shape
income_raw = data['income']
features_raw = data.drop('income', axis = 1)
income = income_raw.apply(lambda x: 0 if x == '<=50K' else 1)
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])
features = pd.get_dummies(features_raw)
#display(features.head(10))

#Now let's try to plot those non-numerical columns, start from workclass
a = features_raw['workclass']
y = income
data1 = pd.concat([a, y], axis = 1)
#grouped = data1['income'].groupby(data1['workclass'])
grouped = data1.groupby(['workclass', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "workclass vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, education_level
a = features_raw['education_level']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['education_level', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "education_level vs income") #, stacked = True)
plt.show()

#Now let's try to plot compare education_level vs. education-num
a = features_raw['education-num']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['education-num', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "education-num vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, marital-status
a = features_raw['marital-status']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['marital-status', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "marital-status vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, occupation
a = features_raw['occupation']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['occupation', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "occupation vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, relationship
a = features_raw['relationship']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['relationship', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "relationship vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, race
a = features_raw['race']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['race', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "race vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, sex
a = features_raw['sex']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['sex', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "sex vs income") #, stacked = True)
plt.show()

#Now let's try to plot those non-numerical columns, native-country
a = features_raw['native-country']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['native-country', 'income'])
print "Here is grouped dataframe:", grouped.size()
#new_group = grouped.size().drop('United-States')
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "native-country vs income") #, stacked = True)
plt.show()

#Figuring the bar chart will show more information,
#here to re-plot the numerical columnes, start from age
a = features_raw['age']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['age', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "age vs income") #, stacked = True)
plt.show()

#here to re-plot the numerical columnes for education-num
a = features_raw['education-num']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['education-num', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "education-num vs income") #, stacked = True)
plt.show()

#here to re-plot the numerical columnes for capital-gain
a = features['capital-gain']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['capital-gain', 'income'])
print "Here is grouped dataframe:", grouped.size()
new_group = grouped.size().drop(0.000000)
print "Here test new group", new_group
my_plot  = new_group.unstack().plot(kind ='bar', title = "capital-gain vs income") #, stacked = True)
plt.show()

#here to re-plot the numerical columnes for capital-loss
a = features['capital-loss']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['capital-loss', 'income'])
print "Here is grouped dataframe:", grouped.size()
new_group = grouped.size().drop(0.000000)
my_plot  = new_group.unstack().plot(kind ='bar', title = "capital-loss vs income") #, stacked = True)
plt.show()

#here to re-plot the numerical columnes for hours-per-week
a = features_raw['hours-per-week']
y = income
data1 = pd.concat([a, y], axis = 1)
grouped = data1.groupby(['hours-per-week', 'income'])
print "Here is grouped dataframe:", grouped.size()
my_plot  = grouped.size().unstack().plot(kind ='bar', title = "hours-per-week vs income") #, stacked = True)
plt.show()
