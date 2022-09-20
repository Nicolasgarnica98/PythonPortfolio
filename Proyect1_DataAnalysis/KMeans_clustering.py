import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Database_connection import data_preprocesing, df_country

#KMeans model generation
class KMeans_model:

    def __init__(self,X):
        self.X = X

    def model_generation(self,n_clusters, seed, save, archive_name):
        self.Kmeans_model = KMeans(n_clusters=n_clusters, random_state=seed)
        self.trained_model = self.Kmeans_model.fit(self.X)
        self.labels = self.trained_model.labels_
        self.k_means_cluster_centers = self.trained_model.cluster_centers_
        if save == 'True':
            saving_route = f'Proyect1_DataAnalysis/Trained models/{archive_name}.npy'
            np.save(saving_route,self.trained_model)
            print(f'Model saved in: {saving_route}')

        return self.labels, self.k_means_cluster_centers

#Graphical analysis of the cluster separation
class graphical_analysis():

    def __init__(self,X,country_names):
        self.X = X
        self.country_names = country_names

    def graph_clusters(self,n_clusters,labels,k_means_cluster_centers):

        def get_anotations(self,num_annotations,country_names):
            any_annotation = False
            annotations =[]
            if num_annotations > 1:
                any_annotation = True
                for i in range(0,num_annotations):
                    country_code = input('Write the country code in uppercase: ')
                    for idx, ccode in enumerate(country_names):
                        if country_code == ccode:
                            annotations.append([self.X[:,0][idx],self.X[:,1][idx],country_code])
            
            print(annotations)

            return any_annotation,annotations
        fig, ax = plt.subplots(1,2,sharey=True,sharex=True)
        fig.set_size_inches(14, 8)
        colors = ["#4EACC5", "#FF9C34", "#4E9A06",'#db0d0d','#970ddb']
        legend_labels = ('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5')
        subplot_index = len(n_clusters)
        for j in range(subplot_index):
            for i in range(0,n_clusters[j]):
                my_members = labels[j] == i
                cluster_center = k_means_cluster_centers[j][i]
                ax[j].plot(self.X[my_members, 0], self.X[my_members, 1], "w", markerfacecolor=colors[i], marker=".",markersize=10)
                ax[j].plot(cluster_center[0], cluster_center[1], marker="o", markerfacecolor=colors[i], markeredgecolor="k", markersize=12, label=legend_labels[i])
                    
            ann_ =  get_anotations(self,num_annotations=6,country_names=self.country_names)
            for w in range(0,len(ann_[1])):
                ax[j].annotate(ann_[1][w][2], xy=(ann_[1][w][0], ann_[1][w][1]),  xycoords='data',
                               textcoords='data', arrowprops=dict(facecolor='black', arrowstyle="->", 
                               connectionstyle="arc3"), horizontalalignment='right', verticalalignment='top',
                               xytext=(ann_[1][w][0]-2, ann_[1][w][1]+3),bbox=dict(boxstyle="Square", fc="w"))
            ax[j].legend()
            ax[j].set_title(f"KMeans (GNP per country \nand Life expectancy), k = {n_clusters[j]}",fontsize=9)
            ax[j].set_xlabel('Log(GNP per country)')
            ax[j].grid('True',linestyle='dashed')
        ax[0].set_ylabel('Life expectancy (years)')
        plt.tight_layout()
        plt.show()

x, y, kmeans_data, CountryName = data_preprocesing(df_country)

KMeans1_results = KMeans_model(kmeans_data).model_generation(n_clusters=3,seed=1111,save='True',archive_name='KMeansModel_K3')
KMeans2_results = KMeans_model(kmeans_data).model_generation(n_clusters=5,seed=1111,save='True',archive_name='KMeansModel_K5')

Labels = [KMeans1_results[0],KMeans2_results[0]]
KCenters = [KMeans1_results[1],KMeans2_results[1]]
n_clusters = [3,5]

results_analysis = graphical_analysis(kmeans_data,CountryName).graph_clusters(n_clusters,Labels,KCenters)
