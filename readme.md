# Projet dashboard OpenClassrooms

Ce projet représente l'intégralité d'un workflow de data science.
Il commence par un nettoyage de données et une modélisation et poursuit jusqu'au déploiement d'une application utilisable par un client.


### Lancement du dashboard

Pour faire fonctionner le dashboard:
  1. dans le dossier 'API', utiliser la commande 'python API.py'
  2. dans le dossier  'dashboard', utiliser la commande 'streamlit run dashboard.py'

### Inspecter et reproduire l'entrainement du modèle

La démarche d'entrainement du modèle elle est observable dans le fichier 'modélisation.ipynb' dans le dossier modélisation
  - le notebook est consultable tel quel
  - cependant pour faire fonctionner le code à l'intérieur il est nécessaire de lancer le script 'kernel_preprocess.py'
    - Celui-ci va télécharger les données (700Mo), les décompresser puis construire les fichiers necessaires au fonctionnement du notebook
