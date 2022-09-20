import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm

#Database connection
conn = mysql.connector.connect( host="localhost",
                                user="root",
                                password="Teutones98*",
                                database = "world"
                              )

mycursor = conn.cursor()

#Dataframe creation from SQL query
query_country = "SELECT * FROM country WHERE NOT IndepYear='null' AND NOT LifeExpectancy='null'"
df_country = pd.read_sql(query_country,conn)

 
#Data loading and pre-processing
def data_preprocesing(df):
    LifeExpectancy_perCountry = np.expand_dims(np.array(df['LifeExpectancy']).flatten(),1)
    GNP_perCountry = np.expand_dims(np.array(df['GNP']),1)
    CountryName = np.expand_dims(np.array(df_country['Name']),1)

    #Liniarization with natural logarithm
    GNP_perCountry = np.log(GNP_perCountry)
    # LifeExpectancy_perCountry = np.log(LifeExpectancy_perCountry)
    kmeans_data = np.hstack((GNP_perCountry,LifeExpectancy_perCountry))

    return GNP_perCountry, LifeExpectancy_perCountry, kmeans_data, CountryName





