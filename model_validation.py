import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from preprocessing import extract_and_preprocess
from model_selection import define_and_select_model

def validate_model(x, y, model, conf_matrix=False, l_curve=False):
    """
    """
    #Selected model score
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("score du modele : {}".format(score))

    #Confusion matrix
    if conf_matrix:
        c_matrix = confusion_matrix(y_test, model.predict(X_test))
        sb.heatmap(c_matrix, annot=True, cmap="Blues")
        plt.savefig("confusion matrix.png")
        plt.close()

    #learning curve
    if l_curve:
        data_size, train_score, _ = learning_curve(model, X_train, y_train, scoring='accuracy', train_sizes = np.linspace(0.2, 1.0, 500), cv=5)
        _, test_score, _ = learning_curve(model, X_test, y_test, scoring='accuracy', train_sizes = np.linspace(0.2, 1.0, 500), cv=5)

        #Score sur le trainset en fonction du nombre de data
        plt.plot(data_size, train_score.mean(axis=1), label="trainset")

        #Score sur le testset en fonction du nombre de data
        plt.plot(data_size, test_score.mean(axis=1), label="testset")
        plt.legend()
        plt.xlabel("Nombre de données utilisée")
        plt.ylabel("Accuracy")
        plt.title("Learning curve du modèle retenu")
        plt.grid()
        plt.show()
    return model

def tune_adaboost_model(x, y):
    """
    """
    parameters = {'n_estimators':[10,50,100,250],
                  'learning_rate':[0.01,0.1]}
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    adaboot_classifier = AdaBoostClassifier()
    grid = GridSearchCV(estimator = adaboot_classifier, param_grid = parameters, cv=5)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    score = model.score(X_test, y_test)
    print("score du modele Adaboost tunné : {} \n".format(score))
    print("parametres du modele Adaboost tunné : {} \n".format(grid.best_params_))

    return model


def predict_client(indiv, model):
    """
    """
    proba_predicted = model.predict_proba(indiv)
    return proba_predicted


if __name__ == "__main__":
    percentage_of_outliers_to_delete = 0.02
    x, y, _ = extract_and_preprocess(percentage_of_outliers_to_delete)
    x, selected_model, selected_model_name = define_and_select_model(x, y)
    print("selected model : {}".format(selected_model_name))
    model = validate_model(x, y, selected_model, conf_matrix=True, l_curve=False)
    tuned_classifier = tune_adaboost_model(x, y)

    #Prediction for a single data
    proba_predicted = model.predict_proba(x)


