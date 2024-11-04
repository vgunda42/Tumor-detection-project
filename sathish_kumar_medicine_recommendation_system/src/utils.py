import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from exception import CustomException

def save_object(file_path, obj):
    """Saves an object to a file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads an object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# def evaluate_models(X_train, y_train, X_test, y_test, models, params):
#     """Evaluates classification models using GridSearchCV and returns a report of accuracy scores."""
#     try:
#         report = {}

#         for model_name, model in models.items():
#             param = params.get(model_name, {})
            
#             gs = GridSearchCV(model, param, cv=3, scoring='accuracy')
#             gs.fit(X_train, y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train, y_train)

#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             # Calculate accuracy scores
#             train_model_score = accuracy_score(y_train, y_train_pred)
#             test_model_score = accuracy_score(y_test, y_test_pred)

#             report[model_name] = test_model_score

#             print(f"Classification report for {model_name}:\n", classification_report(y_test, y_test_pred))

#         return report
    
#     except Exception as e:
#         raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluates classification models using GridSearchCV and returns a report of accuracy scores."""
    try:
        report = {}

        for model_name, model in models.items():
        
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate accuracy scores
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"Classification report for {model_name}:\n", classification_report(y_test, y_test_pred))

        return report
    
    except Exception as e:
        raise CustomException(e, sys)