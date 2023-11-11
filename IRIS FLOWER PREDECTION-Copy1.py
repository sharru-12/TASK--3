#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df_iris = pd.read_csv('IRIS.csv')
df_iris


# In[3]:


df_iris.info()


# In[4]:


df_iris.describe()


# In[5]:


sns.set_style("whitegrid")
sns.pairplot(df_iris,hue="species", height=3);
plt.show()


# In[6]:


df_iris['species'] = df_iris['species'].astype('category')
species_map = list(df_iris['species'].cat.categories)
species_map


# In[7]:


df_iris['species'] = df_iris['species'].cat.codes
df_iris


# In[8]:


# Split data to X and y
X = df_iris[df_iris.columns[:-1]]
y = df_iris[df_iris.columns[-1]]

X.shape, y.shape


# In[9]:


scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

print(X[:2])
print(X_scaled[:2])


# In[10]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape


# In[11]:


# Create an SVC classifier
clf = SVC(kernel='linear', C=1) 

# Fit the classifier on the training data
clf.fit(X_train, y_train)


# In[12]:


# Make predictions on the test data
y_pred = clf.predict(X_test)
y_pred


# In[13]:


# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[14]:


y_in_category_str = [species_map[pred] for pred in y_pred]
y_in_category_str


# In[15]:


import pickle

save_data = {
    'model': clf,
    'scaler': scaler,
    'species_map': species_map,
}
save_data


# In[16]:


file_path = 'iris-svm.pickle'

# Save data to a file using pickle
with open(file_path, 'wb') as file:
    pickle.dump(save_data, file)


# In[17]:


import pickle

file_path = 'iris-svm.pickle'

# Load data from the pickle file
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

loaded_data


# In[ ]:




