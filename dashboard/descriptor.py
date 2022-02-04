import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cette fonction nous permet d'obtenir une liste des features considérés les plus importants
def get_features(sk_id_curr,dataset,explainer):
    shapDF =  pd.DataFrame({'feature':dataset.columns,
                        'shap':explainer.shap_values(dataset.loc[sk_id_curr,:])})
    orderedShapDF = shapDF.iloc[shapDF.shap.abs().sort_values(ascending=False).index,:]
    return orderedShapDF.head(5)['feature'].values




def get_description(featureName,descriptions):
    info = descriptions[descriptions['Row']==featureName]['Description'] 
    try:
        return info.values[0]
    except :
        return 'pas de description disponible'

    
def plot_feature(sk_id_curr,feature,
                 defaultClientsData,
                 paybackClientsData,
                 unscaledData):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.kdeplot(x=feature,
                data=paybackClientsData,
                fill=True,
                label='Clients ayant remboursé',
                ax=ax)
    sns.kdeplot(x=feature,
                data=defaultClientsData,
                fill=True,
                color='red',
                alpha=0.2,
                label='Clients en défaut',
                ax=ax)
    plt.axvline(x=defaultClientsData[feature].mean(),
               color='red',
               label='Position moyenne des crédits en défaut')
    plt.axvline(x=unscaledData.loc[sk_id_curr,[feature]][0],
               color='cyan',
               label=f'Position du client (crédit n° {sk_id_curr})')
    plt.legend()
    plt.suptitle("Comparaison entre client individuel et totalité des clients.");    
    return fig
   

def interpret_client(sk_id_curr,
                     feature,
                     defaultClientsData,
                     unscaledData):
    meanDefault = defaultClientsData[feature].mean()
    stdDefault = defaultClientsData[feature].std()
    clientValue = unscaledData.loc[sk_id_curr,feature]
    position = (clientValue - meanDefault) / stdDefault
    if position < -2:
        phrase = "très largement en dessous de"
    elif position >= -2 and position < -1:
        phrase = 'largement en dessous de'
    elif position >=-1 and position < -0.5:
        phrase = 'en dessous de'
    elif position >= -0.5 and position < 0.5:
        phrase = 'dans'
    elif position >= 0.5 and position < 1:
        phrase = 'au dessus de'
    elif position >= 1 and position < 2:
        phrase = 'largement au dessus de'
    elif position > 2:
        phrase = 'très largement au dessus de'
    return f'Le client est {phrase} la moyenne des clients en défaut.'    
        
        
