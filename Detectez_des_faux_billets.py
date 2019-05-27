#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("billets.csv")
data = pd.DataFrame(data, columns=data.columns)

data.head()


# # Mission 0

# In[3]:


data.shape


# ## Nettoyage

# In[4]:


data.info() 


# <p> Des valeurs sont manquantes / Non attribuées

# In[5]:


print("Les colonnes qui sont NaN :")
i = 0
while i <= 6:
    print(data.columns[i],":", any(pd.isna(data.iloc[:,i])))
    i += 1


# ## Analyses univariées

# ### Describe sur tout le Df

# In[6]:


data_all = data.describe()
data_all


# <p> On remarque que sur toutes les valeurs, seule "margin_low" et "length" possède un écart type important ; Cela nous indique que les valeurs varient fortement (plus fortement que les autres variables)</p>
# <p> Voyons maintenant par type de billet - Vrai et Faux - comment se comporte les variables.</p>

# ### Describe sur les faux billets

# In[7]:


faux = data[data.is_genuine == False].describe()
faux


# <p> Encore une fois les valeurs "margin_low" et "length" se démarquent dans leur écart type </p>
# <p> Voyons si ces écarts types sont sont les mêmes avec les vrais billets </p>

# ### Describe sur les vrais billets

# In[8]:


vrais = data[data.is_genuine == True].describe()
vrais


# <p> Les minimums et maximums des valeurs True et False correspondent bien, nous ne pouvons rien déduire directement <br/>
# (Si le maximum d'un True est inférieur au minimum d'un False alors nous cela aurait pu nous aider) </p>
# 
# <p> Cependant on remarque que cette fois, les écarts types des variables "margin_low" et "length" ne sont pas très éloignés des autres variables. Il semblerait donc que les faux billets se démarquent par une forte variances en margin_low et length </p>

# <p> Il ne semble pas y avoir de valeur aberrantes de première abord ; Voyons avec des graphiques. </p>

# In[9]:


sns.pairplot(data, hue="is_genuine", vars=data.iloc[:,1:])
plt.savefig("Graphiques/analyses_bivariees.png")
plt.show()


# <p> Les colonnes Margin_low et Length se démarquent par leurs grandes différences entre les vrais et faux billets. <br/> Un billet ayant une valeur de 110mm en length est nécessairement faux, ce qui n'est pas le cas pour diagonal par exemple</p>
# <p> Si l'on décidait de se baser uniquement sur ces histogrammes pour détecter si un billet est vrai ou non, alors le programme ne serait pas satisfaisant ; <br/>
#     En effet on peut voir que certains faux billets ont la même taille (en margin_low ou length) que les vrais billets, et inversement. Nous passerions donc à côté de faux billets. </p>

# ### Histogramme Margin_low et Length

# In[59]:


data[data['is_genuine'] == True]['length'].hist(bins = 20, color = 'b', label="True")
data[data['is_genuine'] == False]['length'].hist(bins = 20, color = 'r', label="False")
plt.xlabel('Length')
plt.ylabel("Nombre d'individus")
plt.legend()
plt.show()

data[data['is_genuine'] == True]['margin_low'].hist(bins = 25, color = 'b', label="True")
data[data['is_genuine'] == False]['margin_low'].hist(bins = 25, color = 'r', label="False")
plt.xlabel('Margin_low')
plt.ylabel("Nombre d'individus")
plt.show()


# <p> On voit qu'il y a des extrêmes pour ces valeurs, voyons maintenant si les extrêmes de l'une des valeurs correspondent aux extrêmes de l'autre. </p>

# ### les outliers

# In[10]:


max_margin = data[data['is_genuine'] == True]['margin_low'].max()
max_length = data[data['is_genuine'] == True]['length'].max()

outliners = data[data["is_genuine"] == False]
outliners = outliners[outliners["margin_low"] > max_margin]
outliners = outliners[outliners['length'] > max_length]
print(outliners)


# <p> Les valeurs au dessus de margin_low.max (pour les vrais billets) ne correspondent pas aux valeurs au dessus de length.max (pour les vrais billets ; Nos outliners prennent donc des valeurs indépendantes pour chaque variables </p>

# # Partie 1 : ACP

# In[11]:


from functions2 import *
from sklearn import preprocessing, decomposition


# ## Composantes principales

# In[60]:


n_comp = 5

data_pca = data[["diagonal", "height_left", 
                 "height_right", "margin_low", 
                 "margin_up", "length"]]

data_pca = data_pca.fillna(data_pca.mean())

#On ne prend que les valeurs
X = data_pca.values

#La variable qui nous permettra de colorer nos résultats : 
fakeornot = [data.loc[fake, "is_genuine"] for fake in data.index]

names = data_pca.index

features = data_pca.columns

#Centrage et réduction : 
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

#Calcul des composantes principales : 
pca = decomposition.PCA(n_components = n_comp)
pca = pca.fit(X_scaled)


# ## Eboulis des valeurs propres

# In[13]:


display_scree_plot(pca)


# <p> L'éboulie des valeurs propres nous indique que 5 composantes principales sont nécessaires pour expliquer presque 100% des données. </p>
# 
# <p> Nous apprenons aussi que la composante principale 1 (PC1) explique 47,4 % des informations.</p>
# 
# <p>Notons aussi qu'un plan factoriel (F) correspond à la projection d'une composante principale (PC) sur un axe. <br/>
# F1 correspond donc à la projection de PC1, F2 à la projection de PC2, ect...</p>

# ## Cercle des corrélations

# In[14]:


pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1)], labels = np.array(features))


# <p> Le cercle des corrélations indique que height_left, height_right, margin_up, margin_low, et length sont corrélées pour F1. </p>
# 
# <p> Afin de vérifier cela par les chiffres, regardons le poids de chaque variable sur l'inertie de l'axe :</p>

# ## Poids des variables 

# In[15]:


poids_des_variables = pd.DataFrame(pcs, index=[["F1", "F2", "F3", "F4", "F5"]], 
                                   columns = features)

pds = poids_des_variables

#On transpose la matrice (utile pour plus tard dans le programme)
pds = pds.T
#On ne prend que F1 et F2 (qui nous intéressent)
pds = pds.iloc[:,:2]
pds


# <p> On retrouve bien ici nos colonnes vues précédemment pour F1. </p>

# ## Projection des individus

# In[16]:


X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1)]
                         , illustrative_var = fakeornot,
                        name="Graphiques/Projection_acp.png")


# <p> La projection des individus sur F1 et F2 nous indique deux clusters, les vrais billets, ainsi que les faux. Il semble donc que nous allons pouvoir nous servir de F1 et F2 pour notre programme. (Attention toutefois certains points du Cluster 'Faux' se démarque peu ou pas du cluster 'Vrai', et inversement - Il faudra donc retravailler ce point pour une plus grande précision du programme).</p>   
# 
# <p> Pour la grande majorité des points, un billet faux correspond donc à une grande valeur pour F1, et une faible valeur pour un billet vrai </p>
# 
# <p> Les autres plans factoriels n'apportent pas d'informations supplémentaires utiles pour notre programme </p>

# ## Contribution des individus dans l'inertie totale : 

# In[17]:


#contribution des individus dans l'inertie totale
di = np.sum(X_scaled**2,axis=1)
contrib = pd.DataFrame({'is_genuine':data["is_genuine"],'d_i':di})
contrib.sort_values(by="d_i", ascending = False).head()


# On a ici les individus ayant le plus de poids dans l'inertie totale ; Utile pour les calculs des parties suivantes.

# ## Qualité de la représentation 

# In[18]:


#qualité de représentation des individus -COS2
cos2 = X_projected**2
for j in range(n_comp):
    cos2[:,j] = cos2[:,j]/di
    
qual_rep = pd.DataFrame({'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]})
qual_rep.head()


# <p> La moyenne de l'inertie de chaque point sur l'axe F1 est d'environ 50%. <br/> Cela veut dire qu'en moyenne les individus sont bien représentés. </p>

# In[19]:


#contributions aux axes
eigval = pca.singular_values_**2/len(data)
ctr = X_projected**2
for j in range(n_comp):
    ctr[:,j] = ctr[:,j]/(len(data)*eigval[j])
    
contrib = pd.DataFrame({'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]})
contrib.sort_values(by=['CTR_1'], ascending = False).head()


# <p> La somme de toutes les contributions sur un axe est bien égale à 1 </p>
# <p> Peut-être intéressant de regarder si les individus qui contribues le plus sont les vrais ou les faux ? </p>
# <p> Point intéressant, aucun individu ne prend de trop grande valeur dans l'inertie, on n'a donc pas d'axe fortement influencé par un (ou quelques) individu seulement. </p>

# ## Construction de l'échantillon avec les données de l'ACP

# In[20]:


factorial = pd.DataFrame(X_projected, index=names, columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
data2 = data["is_genuine"]
data2 = pd.DataFrame(data2, index=names, columns=['is_genuine'])
data2 = pd.merge(data2, factorial,  left_index = True, right_index = True, how = "left")
data2.head()


# # Mission 2 : Appliquer un algorithme de Classification

# In[21]:


from sklearn.cluster import KMeans


# ## Algorithme k-means

# In[22]:


kmeans = KMeans(n_clusters=2, random_state=4).fit(data2.iloc[:,1:])

kmeans = pd.DataFrame(kmeans.labels_, index = data.index, columns=["Clusters"])


# In[23]:


data2 = pd.merge(data2, kmeans, left_index=True, right_index=True)
data2.head()


# ## Visualisation sur F1

# In[24]:


clust = [data2.loc[i, "Clusters"] for i in data.index]
display_factorial_planes(X_projected, n_comp, pca, [(0,1)], illustrative_var = clust,
                        name="Graphiques/Projection_kmeans.png")


# <p> Nos deux clusters rejoignent presque intégralement les valeurs vrai/faux que nous avions vu avant </p>
# 

# ## Matrice de confusion et Accuracy

# In[25]:


#Transformons Cluster 1 en True et Cluster 2 en False :
data_confusion = data2.copy()
data_confusion["y_pred"] = np.where(data_confusion["Clusters"] == 1, True, False)

#Calcul de la matrice de confusion :
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(data_confusion["is_genuine"], 
                               data_confusion['y_pred'])

conf_df = {"Vrai":[conf_matrix[1,1],conf_matrix[0,1]], 
           "Faux":[conf_matrix[0,0],conf_matrix[1,0]]}
conf_df = pd.DataFrame(conf_df, index=[["Positif", "Négatif"]])

print(conf_df)
print("\n")
print("L'accuracy est de : ",
      round((conf_matrix[1,1]+conf_matrix[0,0])/len(data_confusion)*100))


# <p> La matrice de confusion indique une accuracy de 95%, ce qui veut dire que le Cluster correspond à hauteur de 95% à la répartition vrai/faux du fichier </p>

# #  Mission 3 : Mise en place du Programme

# In[26]:


import statsmodels.formula.api as smf
import statsmodels.api as sm


# ## Préparation du Df

# In[27]:


#On transforme la colonne is_genuine en 1 et 0 ; 1 = True, 0=False.
data2['is_genuine'] = list(map(int, data['is_genuine'])) 
data['is_genuine'] = list(map(int, data['is_genuine']))


# ## Régression

# ### Données ACP :

# In[28]:


reg_log1 = smf.glm('is_genuine~PC1+PC2',
                   data=data2,
                   family=sm.families.Binomial()).fit()
print(reg_log1.summary())


# <p> On obtient une p_valeur inférieur à 5% avec PC1 et PC2 ; La p_valeur de PC3 dépasse notre seuil de 5%, le paramètre n'est donc pas significativement différent de 0. </p>
# 
# <p> Nous utiliserons donc PC1 et PC2 pour cette régression logistique. (C'est d'ailleurs ce que laissait entendre l'ACP).</p>

# ### Données non ACP :

# In[29]:


reg_log2 = smf.glm('is_genuine~margin_low+margin_up',
                   data=data,
                   family=sm.families.Binomial()).fit()
print(reg_log2.summary())


# <p> Ici seules les colonnes Margin_low et Margin_up sont significatives pour la régression. <br/> On conserve donc ces deux colonnes pour nos prédictions </p>

# ## Définition du Programme

# <p> Pour le programme : <br/>
#     <li><ol>Import du fichier de base, parcourt des différentes colonnes (en vérifiant que les colonnes correspondent bien à celles de notre fichier train)</ol>
#         <ol>Vérifications valeurs manquantes </ol>
#         <ol>Centrage et réduction des données (pour le programme ACP)</ol>
#         <ol>Transformation des données en valeurs PC1 et PC2 (pour le programme ACP) </ol>
#         <ol>Passage dans le .predict de notre régression logistique </ol>
#         <ol>Voir comment est le fichier de sortie, surement un traitement des données à faire ensuite </ol></li>

# ### Programme utilisant les données de l'ACP :

# In[46]:


def billets_acp (df):
    #On récupère les noms de colonnes :
    columns_df = list(df.columns)
    columns_data = list(data[['diagonal','height_left', 
                          'height_right','margin_low', 
                          'margin_up','length']].columns)
    
    #On check si une colonne id existe pour l'utiliser en index par la suite : 
    if any("id" in i for i in columns_df):
        id_df = df['id']
    else:
        id_df = df.index    
    
    #On les compare avec celles de notre fichier de train
    result =  all(elem in columns_df  for elem in columns_data)

    if result == True :
        pass
    else:
        print("Les données fournies ne contiennent pas les colonnes requises.")
        print("Pour que le programme fonctionne, il faut les colonnes suivantes :")
        print(columns_data)
        return None
    #On gère les valeurs manquantes :
    df = df.fillna(df.mean()) 
    
    #On prend les colonnes utiles pour le centrage/réduction : 
    df_cr = df[["diagonal", "height_left", 
                 "height_right", "margin_low", 
                 "margin_up", "length"]]
    
    #On centre et on réduit : 
    from sklearn import preprocessing, decomposition
    df_value = df_cr.values
    df_std_scale = preprocessing.StandardScaler().fit(df_value)
    df_scaled = df_std_scale.transform(df_value)
    
    #On applique maintenant le poids de chaque variable 
    #à la variable elle même
    #Pour obtenir ses coordonnées dans F1 et F2 :
    df_scaled = df_scaled.dot(pds)
    #df_scaled = pca.transform(df_scaled)
    #df_scaled = pd.DataFrame(df_scaled, index=id_df, columns=["PC1", "PC2", "PC3", "PC4", "PC5"])
    #print(df_scaled)
    df_facto = pd.DataFrame(df_scaled, index=id_df, 
                      columns=[["PC1", "PC2"]])
    
    #On applique la prédiction de la régression logistique : 
    df_predict = reg_log1.predict(df_facto)
    
    #On modifie le df retourné pour avoir l'identifiant, la proba et True or False :
    df_predict = pd.DataFrame(df_predict, index=id_df, columns=["Proba"])
    df_predict["is_genuine"] = np.where(df_predict["Proba"] >= 0.50, True, False)
    
    #On enregistre le fichier dans un répertoire :
    #df_final.to_csv("previsions.csv")
    
    #On calcule le nombre de True et de False : 
    true = df_predict[df_predict["is_genuine"] == True].count()
    false = df_predict[df_predict["is_genuine"] == False].count()
    
    #On print le tout : 
    print("Le nombre de True est de : ", true[1])
    print("Le nombre de False est de : ", false[1])
    
    #On save le fichier avec les True uniquement : 
    vrais_billets = df_predict[df_predict["is_genuine"] == True]
    #vrais_billets.to_csv("vrais_billets.csv")
    
    #On return le tout pour l'avoir en sortie de programme :
    return df_predict


# ### Programme sans données de l'ACP :

# In[31]:


def billets (df):
    #On récupère les noms de colonnes :
    columns_df = list(df.columns)
    columns_data = list(data[['diagonal','height_left', 
                          'height_right','margin_low', 
                          'margin_up','length']].columns)
    
    #On check si une colonne id existe pour l'utiliser en index par la suite : 
    if any("id" in i for i in columns_df):
        id_df = df['id']
        id_df = pd.DataFrame(id_df, index=df.index, columns=['id'])
    else:
        id_df = df.index    
    id_df = pd.DataFrame(id_df, index=df.index, columns=['id'])
    
    #On les compare avec celles de notre fichier de train
    result =  all(elem in columns_df  for elem in columns_data)

    if result == True :
        pass
    else:
        print("Les données fournies ne contiennent pas les colonnes requises.")
        print("Pour que le programme fonctionne, il faut les colonnes suivantes :")
        print(columns_data)
        return None
    
    #On gère les valeurs manquantes :
    df = df.fillna(df.mean())
    
    #On applique la prédiction de la régression logistique :
    df_predict = reg_log2.predict(df)
    
    #On modifie le df retourné pour avoir l'identifiant, la proba et True or False :
    import numpy as np
    df_predict = pd.DataFrame(df_predict, index=df.index, columns=["Proba"])
    df_predict = pd.merge(id_df, df_predict, left_index = True, right_index=True)
    df_predict["is_genuine"] = np.where(df_predict["Proba"] >= 0.50, True, False)
    
    #On enregistre le fichier dans un répertoire :
    #df_predict.to_csv("previsions.csv")
    
    #On calcule le nombre de True et de False : 
    true = df_predict[df_predict["is_genuine"] == True].count()
    false = df_predict[df_predict["is_genuine"] == False].count()
    
    #On print le tout : 
    print("Le nombre de True est de : ", true[1])
    print("Le nombre de False est de : ", false[1])
    
    #On save le fichier avec les True uniquement : 
    vrais_billets = df_predict[df_predict["is_genuine"] == True]
    #vrais_billets.to_csv("vrais_billets.csv")
    
    #On return le tout pour l'avoir en sortie de programme :
    return df_predict


# ## Matrice de confusion et Accuracy

# ### Programme ACP :

# In[47]:


#Préparation des données : 
test_data = data.iloc[:,1:]
test_data = billets_acp(test_data)
test_data = test_data['is_genuine']

data_confusion1 = data.copy()
data_confusion1["y_pred"] = test_data

#Calcul de la matrice de confusion :
from sklearn.metrics import confusion_matrix
conf_matrix1 = confusion_matrix(data_confusion1["is_genuine"], 
                               data_confusion1['y_pred'])

conf_df1 = {"Vrai":[conf_matrix1[1,1],conf_matrix1[0,1]], 
           "Faux":[conf_matrix1[0,0],conf_matrix1[1,0]]}
conf_df1 = pd.DataFrame(conf_df1, index=[["Positif", "Négatif"]])

print(conf_df1)
print("\n")
print("L'accuracy est de : ",
      round((conf_matrix1[1,1]+conf_matrix1[0,0])/len(data_confusion1)*100))


# <p> L'accuracy est à 96%, voyons l'accuracy en utilisant les données du df et non de l'ACP </p>

# ### Programme non ACP : 

# In[33]:


#Préparation des données : 
test_data = data.iloc[:,1:]
test_data = billets(test_data)
test_data = test_data['is_genuine']

data_confusion2 = data.copy()
data_confusion2["y_pred"] = test_data

#Calcul de la matrice de confusion :
from sklearn.metrics import confusion_matrix
conf_matrix2 = confusion_matrix(data_confusion2["is_genuine"], 
                               data_confusion2['y_pred'])

conf_df2 = {"Vrai":[conf_matrix2[1,1],conf_matrix2[0,1]], 
           "Faux":[conf_matrix2[0,0],conf_matrix2[1,0]]}
conf_df2 = pd.DataFrame(conf_df2, index=[["Positif", "Négatif"]])

print(conf_df2)
print("\n")
print("L'accuracy est de : ",
      round((conf_matrix2[1,1]+conf_matrix2[0,0])/len(data_confusion2)*100))


# <p> Ici l'accuracy est de 98%, c'est donc le programme utilisant les données du dataframe et non celles de l'ACP que nous utiliserons. </p>

# # Test 

# In[34]:


#Test avec le fichier examples : 
test = pd.read_csv("example.csv")
test


# In[35]:


billets(test)


# In[ ]:




