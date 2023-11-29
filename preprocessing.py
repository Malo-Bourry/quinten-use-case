import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def extract_data():
    """
    Extract the dataset.
    """
    data = pd.read_csv("dataset.csv", sep=';')
    x = data.drop(labels="churn", axis=1)
    y = data["churn"]

    #Deletion of the useless "id_client" feature
    x = x.drop(labels="id_client", axis=1)

    #Binarization of y data.
    y = y.map({'non':0, 'oui':1})
    return x, y, data

def preprocess_categorical_data(x):
    """
    """
    binary_categorical_labels = ["assurance_vie", "banque_principale", "compte_epargne", "compte_courant", "espace_client_web", "espace_client", "PEA", "assurance_auto", "assurance_habitation", "credit_immo", "compte_titres"]
    non_binary_categorical_labels = ["type", "genre", "credit_autres", "cartes_bancaires", "methode_contact", "segment_client", "branche"]
    categorical_labels = binary_categorical_labels

    for labels in enumerate(non_binary_categorical_labels):
        categorical_labels.append(labels[1])
    
    #Imputing of categorical data
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for categorical_features in enumerate(categorical_labels):
        array_to_imput = np.array(x[categorical_features[1]])
        array_to_imput = array_to_imput.reshape(array_to_imput.size, 1)
        array_to_imput = imputer.fit_transform(array_to_imput)
        x[categorical_features[1]] = pd.DataFrame(array_to_imput)
    
    #encoding of categorical data
    encoder = LabelEncoder()
    for labels in enumerate(categorical_labels):
        data_to_encode = np.array(x[labels[1]])    
        data_to_encode = encoder.fit_transform(data_to_encode)
        x[labels[1]] = pd.DataFrame(data_to_encode)
    
    return x

def preprocess_quantitative_features(x):
    """
    """
    n_quantitative_data = 39
    other_quantitative_data = ["anciennete_mois", "agios_6mois", "age"]

    #Imputing of quantitative data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    for i in range(n_quantitative_data):
        array_to_imput = np.array(x["var_{}".format(i)])
        array_to_imput = array_to_imput.reshape(array_to_imput.size, 1)
        array_to_imput = imputer.fit_transform(array_to_imput)
        x["var_{}".format(i)] = pd.DataFrame(array_to_imput)
        
    for labels in enumerate(other_quantitative_data):
        array_to_imput = np.array(x[labels[1]])
        array_to_imput = array_to_imput.reshape(array_to_imput.size, 1)
        array_to_imput = imputer.fit_transform(array_to_imput)
        x[labels[1]] = pd.DataFrame(array_to_imput)

    #Normalization of quantitative data
    scaler = StandardScaler()

    for i in range(n_quantitative_data):
        array_to_imput = np.array(x["var_{}".format(i)])
        array_to_imput = array_to_imput.reshape(array_to_imput.size, 1)
        array_to_imput = scaler.fit_transform(array_to_imput)
        x["var_{}".format(i)] = pd.DataFrame(array_to_imput)

    for labels in enumerate(other_quantitative_data):
        array_to_imput = np.array(x[labels[1]])
        array_to_imput = array_to_imput.reshape(array_to_imput.size, 1)
        array_to_imput = scaler.fit_transform(array_to_imput)
        x[labels[1]] = pd.DataFrame(array_to_imput)

    #"interet_compte_epargne_total" feature normalization
    model = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                          SimpleImputer(missing_values=" ", strategy='most_frequent'),
                          StandardScaler())

    array_to_process = np.array(x["interet_compte_epargne_total"])
    array_to_process = array_to_process.reshape(array_to_process.size, 1)
    preprocessed_data = model.fit_transform(array_to_process)
    preprocessed_data = np.asarray(preprocessed_data, float)
    x["interet_compte_epargne_total"] = pd.DataFrame(preprocessed_data)

    return x

def extract_and_preprocess():
    """
    """
    x, y, data = extract_data()
    x = preprocess_categorical_data(x)
    x = preprocess_quantitative_features(x)
    return x, y, data



if __name__ == "__main__":
    x, y, data = extract_and_preprocess()
