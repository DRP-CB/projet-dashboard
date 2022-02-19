import streamlit as st
import pandas as pd
import joblib
import shap
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
import json
import requests
import numpy as np


datasetPath = "sampleData.csv"
explainerPath = "explainer.pkl"
descriptionsPath = "descriptions.csv"
scalerPath = "scaler.pkl"
url = "https://paulderuta.pythonanywhere.com/pred"

# Les risques crédits inférieurs à cette valeur sont acceptés
seuilattribution = 0.69

# Chargement des données
@st.cache(persist=True)
def load_data():
    dataWithTarget = pd.read_csv(datasetPath, index_col="SK_ID_CURR")

    scaler = joblib.load(scalerPath)
    unscaledData = pd.DataFrame(scaler.inverse_transform(dataWithTarget))
    unscaledData.index = dataWithTarget.index
    unscaledData.columns = dataWithTarget.columns

    data = dataWithTarget.loc[:, dataWithTarget.columns != "TARGET"]

    defaultClientsData = unscaledData[unscaledData["TARGET"] == 1]
    paybackClientsData = unscaledData[unscaledData["TARGET"] == 0]
    unscaledData.rename(columns={"TARGET": "Etat du Crédit"}, inplace=True)
    unscaledData["Etat du Crédit"].replace(0, "Remboursé", inplace=True)
    unscaledData["Etat du Crédit"].replace(1, "Défaut", inplace=True)

    descriptions = pd.read_csv(descriptionsPath)

    return data, unscaledData, defaultClientsData, paybackClientsData, descriptions


@st.cache(allow_output_mutation=True)
def load_explainer():
    shap.initjs()
    explainer = joblib.load(explainerPath)
    return explainer


# la requette POST à l'API
@st.cache(persist=True)
def get_prediction(ID, url, data):
    payload = json.dumps(dict(data.loc[ID, :]))
    return requests.post(url, json=payload).json()


# Cette fonction permet d'avoir des nombres plus lisibles dans les graphiques
@st.cache(persist=True)
def readable_number(number, tick_number):
    if number < 0:
        return f"{np.round(np.abs(number / 365), 2)} ans"
    if number < 1000000:
        return np.round(number, 2)
    elif number >= 1000000 and number < 1000000000:
        str_number = str(np.round(number / 1000000, 2))
        return f"{str_number} Million(s)"
    elif number >= 1000000000:
        str_number = str(np.round(number / 1000000000, 2))
        return f"{str_number} Milliard(s)"


# convertit l'age des clients de jours en années pour les graphiques
@st.cache(persist=True)
def readable_age(number, tick_number):
    return f"{np.round(number / 365, 2)} ans"


# graphique permettant de positionner un individu par rapport à l'ensemble des clients
def plot_feature(
    sk_id_curr, feature, defaultClientsData, paybackClientsData, unscaledData
):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        x=feature,
        data=paybackClientsData,
        fill=True,
        label="Clients ayant remboursé",
        ax=ax,
    )
    if feature == "DAYS_BIRTH":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(readable_age))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(readable_number))

    sns.kdeplot(
        x=feature,
        data=defaultClientsData,
        fill=True,
        color="red",
        alpha=0.2,
        label="Clients en défaut",
        ax=ax,
    )
    plt.locator_params(axis="x", nbins=8)
    plt.axvline(
        x=defaultClientsData[feature].mean(),
        color="red",
        label="Position moyenne des crédits en défaut",
    )
    plt.axvline(
        x=unscaledData.loc[sk_id_curr, [feature]][0],
        color="cyan",
        label=f"Position du client (crédit n° {sk_id_curr})",
    )
    plt.legend()
    plt.suptitle("Comparaison entre client individuel et totalité des clients.")
    return fig


# permet de récupérer les features à expliquer pour chaque client, les plus importants
def get_features(sk_id_curr, dataset, explainer):
    shapDF = pd.DataFrame(
        {
            "feature": dataset.columns,
            "shap": explainer.shap_values(dataset.loc[sk_id_curr, :]),
        }
    )
    orderedShapDF = shapDF.iloc[shapDF.shap.abs().sort_values(ascending=False).index, :]
    return orderedShapDF.head(5)["feature"].values


# Explicite les features par une descritpion lorsque celle-ci est disponible.
def get_description(featureName, descriptions):
    info = descriptions[descriptions["Row"] == featureName]["Description"]
    try:
        return info.values[0]
    except:
        return "pas de description disponible"


# calcule un score Z pour aider à interpréter le positionnement d'un individu par rapport à la masse.
def interpret_client(sk_id_curr, feature, defaultClientsData, unscaledData):
    meanDefault = defaultClientsData[feature].mean()
    stdDefault = defaultClientsData[feature].std()
    clientValue = unscaledData.loc[sk_id_curr, feature]
    position = (clientValue - meanDefault) / stdDefault
    if position < -2:
        phrase = "très largement en dessous de"
    elif position >= -2 and position < -1:
        phrase = "largement en dessous de"
    elif position >= -1 and position < -0.5:
        phrase = "en dessous de"
    elif position >= -0.5 and position < 0.5:
        phrase = "dans"
    elif position >= 0.5 and position < 1:
        phrase = "au dessus de"
    elif position >= 1 and position < 2:
        phrase = "largement au dessus de"
    elif position > 2:
        phrase = "très largement au dessus de"
    return f"Le client est {phrase} la moyenne des clients en défaut."


data, unscaledData, defaultClientsData, paybackClientsData, descriptions = load_data()


panneau1 = "Attribution de crédit"
panneau2 = "Analyse de données"

usecase = st.sidebar.radio("Fonctionnalité : ", [panneau1, panneau2])
if usecase == panneau1:

    st.write(
        "# Attribution de crédit \n Sélectionner / taper un ID crédit pour obtenir une recommandation et l'interpréter."
    )
    ID = st.selectbox("ID du crédit", options=data.index)

    if st.button("Prédire"):
        pred = get_prediction(ID, url, data)

        st.write("###### Recommandation:")
        if pred > seuilattribution:

            original_title = '<p style="font-family:Courier; color:Red; font-size: 40px;">Défavorable</p>'
            st.markdown(original_title, unsafe_allow_html=True)

        elif pred < seuilattribution:

            original_title = '<p style="font-family:Courier; color:Green; font-size: 40px;">Favorable</p>'
            st.markdown(original_title, unsafe_allow_html=True)

        explainer = load_explainer()
        fig, ax = plt.subplots(1, 1)
        ax = shap.bar_plot(
            explainer.shap_values(data.loc[ID, :]),
            feature_names=data.columns,
            max_display=5,
        )
        plt.suptitle(
            "Facteurs de décision \n (bleu = en faveur, rouge = à l'encontre)",
            fontsize=15,
        )
        st.pyplot(fig)

        st.write("### Détail des facteurs :")
        facteurs = get_features(ID, data, explainer)
        for facteur in facteurs:
            with st.expander(facteur):
                st.write(f"{get_description(facteur,descriptions)}")
                fig = plot_feature(
                    ID, facteur, defaultClientsData, paybackClientsData, unscaledData
                )
                st.pyplot(fig)
                position = interpret_client(
                    ID, facteur, defaultClientsData, unscaledData
                )
                st.text(f"{position}")
        with st.expander("Afficher toutes les données"):
            st.table(data[data.index == ID].T)


elif usecase == panneau2:
    # les facteurs utilisés pour sélectionner des clients pour l'analyse
    # les 10 features les plus importants sur l'interprétation globale du modèle
    nonBooleanFeatures = ["AMT_GOODS_PRICE", "AMT_CREDIT", "EXT_SOURCE_2"]

    minsAndMaxesFeatures = [(40500, 4050000), (45000, 4050000), (0.0, 1.0)]

    booleanFeatures = [
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_18",
        "ORGANIZATION_TYPE_Transport: type 3",
        "ORGANIZATION_TYPE_Realtor",
        "ORGANIZATION_TYPE_Construction",
    ]

    st.write("# Analyse de données")
    st.write(
        "Sélectionner la totalité ou un sous ensemble de clients pour \
              observer la relation entre des variables et le risque crédit."
    )
    with st.form(key="dataAnalysis"):
        st.write("#### Filtrer les clients sur les critères suivants :")
        # Chaque checkbox / slider ajoute un masque de sélection
        masks = []
        for feature, bounds in zip(nonBooleanFeatures, minsAndMaxesFeatures):
            value = st.slider(
                feature,
                bounds[0],
                bounds[1],
                (bounds[0], bounds[1]),
                help=get_description(feature, descriptions),
            )
            masks.append(
                (unscaledData[feature] >= value[0])
                & (unscaledData[feature] <= value[1])
            )

        for feature in booleanFeatures:
            if st.checkbox(feature, help=get_description(feature, descriptions)):
                value = 1
            else:
                value = 0
            masks.append(unscaledData[feature] == value)

        # Une fois tous les masques créés nous les combinons pour inclure seulement
        # les données répondant aux critères de chaque slider / checkbox
        combinedMask = np.array(masks).all(axis=0)
        selectedData = unscaledData[combinedMask]
        st.write(
            "#### Pour les clients sélectionnés, analyser les variables suivantes :"
        )
        features = st.multiselect(
            label="Choisir une ou plusieurs variables.", options=data.columns,
        )
        # La visualisation s'adapte en fonction du nombre de variables choisies
        # 1 = histogramme
        # 2 = scatterplot
        # 3 et + = matrice de scatterplots
        submit_button = st.form_submit_button(label="Visualiser")
        if not features:
            pass
        elif selectedData.shape[0] == 0:
            st.write("### Aucune donnée disponible avec les filtres selectionnés. \n #### Essayez une sélection plus générale.")
        elif len(features) < 2:
            fig = px.histogram(
                selectedData,
                x=features,
                template="seaborn",
                color="Etat du Crédit",
                title=f"Distribution de la variable {features[0]} ",
            )
            st.plotly_chart(fig)
        elif len(features) == 2:
            fig = px.scatter(
                x=features[0],
                y=features[1],
                data_frame=selectedData,
                hover_name=selectedData.index,
                opacity=0.8,
                template="seaborn",
                title=f"Relation entre {features[0]} et {features[1]} ",
                color="Etat du Crédit",
            )
            st.plotly_chart(fig)
        elif len(features) > 2:
            fig = px.scatter_matrix(
                selectedData, dimensions=features, color="Etat du Crédit"
            )
            st.plotly_chart(fig)
