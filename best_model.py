import pickle
# from parameters import classification_grid_parameters, regression_grid_parameters
from models_parameters import classification_grid_parameters, regression_grid_parameters

from emotion_recognition import EmotionRecognizer

emotions = ["neutral", "calm", "happy",
            "sad", "angry", "fear",
            "disgust", "ps", "boredom"]
# number of parallel jobs during the grid search
n_jobs = 4

best_estimators = []

# for model, params in classification_grid_parameters.items():
#     if model.__class__.__name__ == "KNeighborsClassifier":
#         # in case of a K-Nearest neighbors algorithm
#         # set number of neighbors to the length of emotions
#         params['n_neighbors'] = [len(emotions)]
#     print(model)
#     # we send the model to train
#     d = EmotionRecognizer(model, emotions=emotions)
#     d.load_data()
#     best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
#     best_estimators.append((best_estimator, best_params, cv_best_score))
#     print(
#         f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation accuracy score!")
#
# best_estimators


for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params['n_neighbors'] = [len(emotions)]
    print(model)
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation MAE score!")

