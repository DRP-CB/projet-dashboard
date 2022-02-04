import streamlit as st
import pandas as pd
import joblib
from request_api import send_request
from descriptor import get_features, get_description, plot_feature, interpret_client
import shap
from matplotlib import pyplot as plt
import plotly.express as px

# st.set_option('deprecation.showPyplotGlobalUse', False)

# variables

dataset = "sampleData.csv"
url = "http://localhost:5000/pred"
seuilattribution = 0.71


@st.cache(persist=True)
def load_data():
    dataWithTarget = pd.read_csv(dataset, index_col="SK_ID_CURR")

    scaler = joblib.load("scaler.pkl")
    unscaledData = pd.DataFrame(scaler.inverse_transform(dataWithTarget))
    unscaledData.index = dataWithTarget.index
    unscaledData.columns = dataWithTarget.columns

    data = dataWithTarget.loc[:, dataWithTarget.columns != "TARGET"]

    defaultClientsData = unscaledData[unscaledData["TARGET"] == 1]
    paybackClientsData = unscaledData[unscaledData["TARGET"] == 0]
    descriptions = pd.read_csv("descriptions.csv")

    # data = data.sample(2000)

    return data, unscaledData, defaultClientsData, paybackClientsData, descriptions


@st.cache(persist=True)
def load_explainer():
    shap.initjs()
    explainer = joblib.load("explainer.pkl")
    return explainer


@st.cache(persist=True)
def get_prediction(ID, url, data):
    pred = send_request(ID, url, data)
    return pred


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
            st.write(f"##### {facteur}\n {get_description(facteur,descriptions)}")
            fig = plot_feature(
                ID, facteur, defaultClientsData, paybackClientsData, unscaledData
            )
            st.pyplot(fig)
            position = interpret_client(ID, facteur, defaultClientsData, unscaledData)
            st.text(f"{position}")

elif usecase == panneau2:
    st.write("# Analyse de données\n ")

    with st.form(key="dataAnalysis"):
        values = st.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
        st.write("Values:", values)
        # st.write('###### Analyser et comparer la/les variables selectionnées')
        features = st.multiselect(
            label="Analyser et comparer la/les variables selectionnées",
            options=data.columns,
        )
        submit_button = st.form_submit_button(label="Visualiser")
    if len(features) < 2:
        fig = px.histogram(data, x=features, template="seaborn")
        st.plotly_chart(fig)
    elif len(features) == 2:
        fig = px.scatter(
            x=features[0],
            y=features[1],
            data_frame=data,
            hover_name=data.index,
            opacity=0.8,
            template="seaborn",
            title=f"Relation entre {features[0]} et {features[1]} ",
        )
        st.plotly_chart(fig)
    elif len(features) > 2:
        fig = px.scatter_matrix(data, dimensions=features)
        st.plotly_chart(fig)
