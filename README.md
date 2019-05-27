# Projet_6
Le but du projet est de prédire si un billet est vrai ou non.

Pour cela un fichier de 170 billets annotés Vrai ou Faux nous est donné. Le fichier contient aussi la diagonal, height_left/right, margin_low/up, et length de chaque billet.

Etape : 

1) Nettoyage des données
2) Analyses univariées et bivariées
3) ACP
4) Kmeans
5) Régression logistique (ici deux régressions sont proposées, une avec les données de l'ACP -F1 et F2- l'autre avec les données du fichier -Margin up et low-)
6)Construction des programmes (un pour chaque régression logistique)
7) Matrice de confusion de chacun des programmes, le programme choisi est celui qui possède la meilleure accuracy

Le fichier ne comportant que très peu d'individus, le choix a été fait de ne pas fractionner le DataFrame en Train/Test.

Le fichier du programme se nomme 'Détectez_des_faux_billets"

Le fichier function_2 provient originellement des cours OpenClassrooms, comme le laisse entendre le "2", je l'ai modifié au fil des projets.
