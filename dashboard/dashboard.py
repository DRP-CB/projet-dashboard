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

# variables

datasetPath = "sampleData.csv"
explainerPath = "explainer.pkl"
descriptionsPath = "descriptions.csv"
scalerPath = "scaler.pkl"
url = "http://localhost:5000/pred"

seuilattribution = 0.73


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

    # data = data.sample(2000)

    return data, unscaledData, defaultClientsData, paybackClientsData, descriptions


@st.cache(persist=True)
def load_explainer():
    shap.initjs()
    explainer = joblib.load(explainerPath)
    return explainer


@st.cache(persist=True)
def get_prediction(ID, url, data):
    payload = json.dumps(dict(data.loc[ID, :]))
    return requests.post(url, json=payload).json()


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


@st.cache(persist=True)
def readable_age(number, tick_number):
    return f"{np.round(number / 365, 2)} ans"


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


def get_features(sk_id_curr, dataset, explainer):
    shapDF = pd.DataFrame(
        {
            "feature": dataset.columns,
            "shap": explainer.shap_values(dataset.loc[sk_id_curr, :]),
        }
    )
    orderedShapDF = shapDF.iloc[shapDF.shap.abs().sort_values(ascending=False).index, :]
    return orderedShapDF.head(5)["feature"].values


def get_description(featureName, descriptions):
    info = descriptions[descriptions["Row"] == featureName]["Description"]
    try:
        return info.values[0]
    except:
        return "pas de description disponible"


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
        with st.expander("Afficher toutes les données"):
            st.table(data[data.index == ID])

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


elif usecase == panneau2:

    nonBooleanFeatures = ["AMT_GOODS_PRICE", "AMT_CREDIT", "EXT_SOURCE_2"]

    minsAndMaxesFeatures = [(40500, 4050000), (45000, 4050000), (0.0, 1.0)]

    booleanFeatures = [
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_18",
        "ORGANIZATION_TYPE_Transport: type 3",
        "ORGANIZATION_TYPE_Trade: type 2",
        "FLAG_DOCUMENT_13",
        "OCCUPATION_TYPE_Accountants",
        "OCCUPATION_TYPE_High skill tech staff",
    ]

    st.write("# Analyse de données")
    st.write(
        "Sélectionner la totalité ou un sous ensemble de clients pour \
              observer la relation entre des variables et le risque crédit."
    )
    with st.form(key="dataAnalysis"):
        st.write("#### Filtrer les clients sur les critères suivants :")

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

        combinedMask = np.array(masks).all(axis=0)
        selectedData = unscaledData[combinedMask]
        st.write(
            "#### Pour les clients sélectionnés, analyser les variables suivantes :"
        )
        features = st.multiselect(
            label="Choisir une ou plusieurs variables.", options=data.columns,
        )
        submit_button = st.form_submit_button(label="Visualiser")
        if not features:
            pass
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
