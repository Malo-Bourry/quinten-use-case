
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocessing import extract_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def select_features(x, y, classifier):
    """
    Selects the best features for a model with the method SelectFromModel.
    """
    selector = SelectFromModel(classifier)
    selector.fit(x, y)
    boolean_value_features = selector.get_support()
    x_new = x.copy()

    for feature in enumerate(x.columns):
        #Selection of the best features according the SelectFromModel selector
        if not boolean_value_features[feature[0]]:
            x_new = x_new.drop(labels=feature[1], axis=1)
    return x_new, boolean_value_features

def train_score_model(x, y, classifier):
    """
    Computes the score (accuracy metric) on the basis of a cross validation of a train set.
    """
    #Split of the dataset between trainset and testset
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    #Classifier training
    classifier.fit(x_train, y_train)

    #Score computing (accuracy metric for the score)
    score = cross_val_score(classifier, x_test, y_test, cv=5, scoring='accuracy').mean()
    return score

def select_model(x, y, classifiers, classifiers_names):
    """
    Compares the score of several classifiers stored in the list classifiers.
    """    
    classifiers_scores = []
    for classifier in classifiers:
        x_new, _ = select_features(x, y, classifier)
        score = train_score_model(x_new, y, classifier)
        classifiers_scores.append(score)

    for i in range(len(classifiers_names)):
        print("Modele : {}, score : {} \n".format(classifiers_names[i], classifiers_scores[i]))
    
    #Recovery of the best model according the accuracy computing
    selected_model = classifiers[np.argmax(classifiers_scores)]
    selected_model_name = classifiers_names[np.argmax(classifiers_scores)]
    
    #Recovery of the features selection for the selected model
    x_new, selected_features = select_features(x, y, selected_model)
    return x_new, selected_model, selected_model_name, selected_features

def define_and_select_model(x, y):
    """
    Defines the classifiers to be compared and compares their score with select_model.
    """
    classifiers = [LogisticRegression(), SGDClassifier(), AdaBoostClassifier(), RandomForestClassifier()]
    classifiers_names = ["RegLog", "SGD", "AdaBoost", "RandomForest"]
    x, selected_model, selected_model_name, selected_features = select_model(x, y, classifiers, classifiers_names)
    return x, selected_model, selected_model_name, selected_features

if __name__ == "__main__":
    percentage_of_outliers_to_delete = 0.02
    x, y, _ = extract_and_preprocess(percentage_of_outliers_to_delete)
    x, _, selected_model_name, _ = define_and_select_model(x, y)    
    print("selected model : {}".format(selected_model_name))



