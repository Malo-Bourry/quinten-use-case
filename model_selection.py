
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
    """
    selector = SelectFromModel(classifier)
    selector.fit(x, y)
    boolean_value_features = selector.get_support()
    x_new = x.copy()

    for feature in enumerate(x.columns):
        if not boolean_value_features[feature[0]]:
            x_new = x_new.drop(labels=feature[1], axis=1)
    return x_new

def train_score_model(x, y, classifier):
    """
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    classifier.fit(x_train, y_train)
    score = cross_val_score(classifier, x_test, y_test, cv=5, scoring='accuracy').mean()
    return score

def select_model(x, y, classifiers, classifiers_names):
    """
    """    
    classifiers_scores = []
    for classifier in classifiers:
        x = select_features(x, y, classifier)
        classifiers_scores.append(train_score_model(x, y, classifier))

    for i in range(len(classifiers_names)):
        print("Modele : ", classifiers_names[i], ", score : ", classifiers_scores[i], "\n")
    selected_model = classifiers[np.argmax(classifiers_scores)]
    return selected_model

if __name__ == "__main__":
    x, y, _ = extract_and_preprocess()
    classifiers = [LogisticRegression(), SGDClassifier(),AdaBoostClassifier(), RandomForestClassifier()]
    classifiers_names = ["RegLog", "SGD", "AdaBoost", "RandomForest"]
    selected_model = select_model(x, y, classifiers, classifiers_names)



