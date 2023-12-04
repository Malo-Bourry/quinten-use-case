from model_validation import construct_adaboost_model
from preprocessing import extract_and_preprocess

def predict(x, model, selected_features):
    """
    Returns the probability for each data of belonging to each target class (not churning or churning).
    """
    x = x.loc[:, selected_features]
    result = model.predict_proba(x)
    return result


if __name__ == "__main__":
    percentage_of_outliers_to_delete = 0.02

    #Model training
    model, selected_features = construct_adaboost_model(percentage_of_outliers_to_delete)    
    
    #Information about new client
    x, _, _ = extract_and_preprocess(0)
    x = x[:3]

    #Prediction
    print("pourcentage de chance que le(s) client(s) étudié(s) quitte(nt) la banque : {}".format(predict(x, model, selected_features)[:, 1]))