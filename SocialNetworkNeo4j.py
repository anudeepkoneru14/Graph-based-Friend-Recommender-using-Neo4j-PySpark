#!/usr/bin/env python
# coding: utf-8

# ### Export the data to Python (Jupyter Notebook + PySpark)

# In[1]:


# get_ipython().system('pip install neo4j pandas')


# In[2]:


from neo4j import GraphDatabase
import pandas as pd

# Fill these with your Aura/Desktop credentials
uri = "bolt://localhost:7687"  # e.g., bolt://localhost:7687 or Neo4j Aura bolt URL
username = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(username, password))


# In[3]:


def fetch_friend_edges(tx):
    query = """
    MATCH (u1:User)-[:FRIEND]-(u2:User)
    WHERE u1.id < u2.id
    RETURN u1.id AS user1, u2.id AS user2
    """
    return list(tx.run(query))

with driver.session() as session:
    edges = session.read_transaction(fetch_friend_edges)

df = pd.DataFrame(edges)
df.drop_duplicates(inplace=True)
df.head()


# In[4]:


def fetch_users(tx):
    query = """
    MATCH (u:User)
    RETURN u.id AS id, u.age AS age, u.location AS location, u.interests AS interests
    """
    return list(tx.run(query))

with driver.session() as session:
    users = session.read_transaction(fetch_users)

users_df = pd.DataFrame(users)
users_df.head()


# In[5]:


df.to_csv("friend_edges.csv", index=False)
users_df.to_csv("user_attributes.csv", index=False)


# ### Prepare Data for Link Prediction

# In[6]:


df = pd.DataFrame(edges, columns=["user1", "user2"])


# In[7]:


df.drop_duplicates(inplace=True)
df.head()


# In[8]:


users = list(set(df['user1'].tolist() + df['user2'].tolist()))


# In[9]:


users


# In[10]:


import random

# Set of existing friend pairs for lookup
positive_set = set(tuple(sorted([a, b])) for a, b in zip(df['user1'], df['user2']))

# Generate negative pairs
neg_samples = set()
while len(neg_samples) < len(positive_set):
    u1, u2 = random.sample(users, 2)
    pair = tuple(sorted([u1, u2]))
    if pair not in positive_set:
        neg_samples.add(pair)

# Create DataFrame for negative samples
neg_df = pd.DataFrame(list(neg_samples), columns=['user1', 'user2'])
neg_df['label'] = 0


# In[11]:


df['label'] = 1
all_df = pd.concat([df, neg_df], ignore_index=True)
all_df = all_df.sample(frac=1).reset_index(drop=True)  # shuffle the data
all_df.head()


# ### Train PySpark ML Pipeline

# In[12]:


import os
os.environ["PYSPARK_PYTHON"] = "python"


# In[ ]:





# In[13]:


# get_ipython().system('pip install pyspark')


# In[14]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FriendLinkPrediction").getOrCreate()


# In[15]:


spark_df = spark.createDataFrame(all_df)
spark_df.printSchema()
spark_df.show(5)


# In[16]:


from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Convert user1 and user2 to numeric indices
indexer1 = StringIndexer(inputCol="user1", outputCol="user1_index")
indexer2 = StringIndexer(inputCol="user2", outputCol="user2_index")

# Assemble into a single feature vector
assembler = VectorAssembler(inputCols=["user1_index", "user2_index"], outputCol="features")

# Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Pipeline
pipeline = Pipeline(stages=[indexer1, indexer2, assembler, lr])


# In[17]:


model = pipeline.fit(spark_df)


# In[18]:


predictions = model.transform(spark_df)
predictions.select("user1", "user2", "label", "prediction", "probability").show(10)


# ### Export results and Streamlit Dashboard

# In[19]:


predictions_pd = predictions.select("user1", "user2", "label", "prediction", "probability").toPandas()
predictions_pd['prob_score'] = predictions_pd['probability'].apply(lambda x: x[1])  # probability of label=1
predictions_pd.to_csv("friend_recommendations.csv", index=False)


# In[21]:


# get_ipython().system('pip install streamlit')


# In[22]:


import streamlit as st
import pandas as pd

# Load prediction data
df = pd.read_csv("friend_recommendations.csv")

# Sort by highest recommendation probability
df = df.sort_values(by="prob_score", ascending=False)

st.title("🤝 Friend Recommendation System")
st.markdown("This app shows predicted friend recommendations based on user interactions.")

# User selection
users = sorted(df['user1'].unique())
selected_user = st.selectbox("Select a User:", users)

# Filter for top recommendations for this user
recommendations = df[(df['user1'] == selected_user) & (df['prediction'] == 1)]
recommendations = recommendations[['user2', 'prob_score']].sort_values(by='prob_score', ascending=False)

st.subheader(f"Top Recommended Friends for User {selected_user}")
st.dataframe(recommendations.head(10))

# Optional: Show false negatives or interesting patterns
st.markdown("----")
show_all = st.checkbox("Show all predictions?")
if show_all:
    st.dataframe(df.head(100))


# In[ ]:




