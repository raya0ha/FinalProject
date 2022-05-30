from utils import load_data, extract_feature, AVAILABLE_EMOTIONS
# from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv

from sklearn.metrics import accuracy_score, make_scorer, fbeta_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as pl
from time import time
from utils import get_audio_config, load_data
import numpy as np
import tqdm
import os
import random
import pandas as pd


class EmotionRecognizer:
    """A class for training, testing and predicting emotions based on
    speech's features that are extracted and fed into `sklearn` or `keras` model"""

    def __init__(self, model=None, **kwargs):
        """
        Params:
            model (sklearn model): the model used to detect emotions. If `model` is None, then self.determine_best_model()
                will be automatically called
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS_TESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom'.
                Default is ["sad", "neutral", "happy"].
            tess_ravdess (bool): whether to use TESS & RAVDESS Speech datasets, default is True
            emodb (bool): whether to use EMO-DB Speech dataset, default is True,
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True
            tess_ravdess_name (str): the name of the output CSV file for TESS&RAVDESS dataset, default is "tess_ravdess.csv"
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv"
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv"
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram )
            classification (bool): whether to use classification or regression, default is True
            balance (bool): whether to balance the dataset ( both training and testing ), default is True
            verbose (bool/int): whether to print messages on certain tasks, default is 1
        Note that when `tess_ravdess`, `emodb` and `custom_db` are set to `False`, `tess_ravdess` will be set to True
        automatically.
        """
        # emotions
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        # make sure that there are only available emotions
        self._verify_emotions()
        # audio config
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        # datasets
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True

        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.verbose = kwargs.get("verbose", 1)
        # boolean attributes
        self.data_loaded = False
        self.model_trained = False

        # model

        self.model = model

    def _verify_emotions(self):
        """
        This method makes sure that emotions passed in parameters are valid.
        """
        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."

    def get_best_estimators(self):
        """Loads estimators from grid files and returns them"""
        return get_best_estimators(self.classification)

    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified
        """
        if not self.data_loaded:
            result = load_data()
            self.X_train = result[0]
            self.X_test = result[1]
            self.y_train = result[2]
            self.y_test = result[3]
            

    def train(self, verbose=1):
        """
        Train the model, if data isn't loaded, it 'll be loaded automatically
        """
        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):
        """
        given an `audio_path`, this method extracts the features
        and predicts the emotion
        """
        feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):
        """
        Predicts the probability of each emotion.
        """
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")

    def grid_search(self, params, n_jobs=2, verbose=1):
        """
        Performs GridSearchCV on `params` passed on the `self.model`
        And returns the tuple: (best_estimator, best_params, best_score).
        """
        score = accuracy_score if self.classification else mean_absolute_error
        grid = GridSearchCV(estimator=self.model, param_grid=params, scoring=make_scorer(score),
                            n_jobs=n_jobs, verbose=verbose, cv=3)
        grid_result = grid.fit(self.X_train, self.y_train)
        return grid_result.best_estimator_, grid_result.best_params_, grid_result.best_score_

    def determine_best_model(self):
        """
        Loads best estimators and determine which is best for test data,
        and then set it to `self.model`.
        In case of regression, the metric used is MSE and accuracy for classification.
        Note that the execution of this method may take several minutes due
        to training all estimators (stored in `grid` folder) for determining the best possible one.
        """
        if not self.data_loaded:
            self.load_data()

        # loads estimators
        estimators = self.get_best_estimators()

        result = []

        if self.verbose:
            estimators = tqdm.tqdm(estimators)

        for estimator, params, cv_score in estimators:
            if self.verbose:
                estimators.set_description(f"Evaluating {estimator.__class__.__name__}")
            detector = EmotionRecognizer(estimator, emotions=self.emotions, tess_ravdess=self.tess_ravdess,
                                         emodb=self.emodb, custom_db=self.custom_db, classification=self.classification,
                                         features=self.features, balance=self.balance)
            # data already loaded
            detector.X_train = self.X_train
            detector.X_test = self.X_test
            detector.y_train = self.y_train
            detector.y_test = self.y_test
            detector.data_loaded = True
            # train the model
            detector.train(verbose=0)
            # get test accuracy
            accuracy = detector.test_score()
            # append to result
            result.append((detector.model, accuracy))

        # sort the result
        # regression: best is the lower, not the higher
        # classification: best is higher, not the lower
        result = sorted(result, key=lambda item: item[1], reverse=self.classification)
        best_estimator = result[0][0]
        accuracy = result[0][1]
        self.model = best_estimator
        self.model_trained = True
        if self.verbose:
            if self.classification:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy * 100:.3f}% test accuracy")
            else:
                print(
                    f"[+] Best model determined: {self.model.__class__.__name__} with {accuracy:.5f} mean absolute error")

    def test_score(self):
        """
        Calculates score on testing data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        y_pred = self.model.predict(self.X_test)
        if self.classification:
            return accuracy_score(y_true=self.y_test, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_test, y_pred=y_pred)

    def train_score(self):
        """
        Calculates accuracy score on training data
        if `self.classification` is True, the metric used is accuracy,
        Mean-Squared-Error is used otherwise (regression)
        """
        y_pred = self.model.predict(self.X_train)
        if self.classification:
            return accuracy_score(y_true=self.y_train, y_pred=y_pred)
        else:
            return mean_squared_error(y_true=self.y_train, y_pred=y_pred)

    def train_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_train)
        return fbeta_score(self.y_train, y_pred, beta, average='micro')

    def test_fbeta_score(self, beta):
        y_pred = self.model.predict(self.X_test)
        return fbeta_score(self.y_test, y_pred, beta, average='micro')

    def confusion_matrix(self, percentage=True, labeled=True):
        """
        Computes confusion matrix to evaluate the test accuracy of the classification
        and returns it as numpy matrix or pandas dataframe (depends on params).
        params:
            percentage (bool): whether to use percentage instead of number of samples, default is True.
            labeled (bool): whether to label the columns and indexes in the dataframe.
        """
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict(self.X_test)
        matrix = confusion_matrix(self.y_test, y_pred, labels=self.emotions).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[f"true_{e}" for e in self.emotions],
                                  columns=[f"predicted_{e}" for e in self.emotions])
        return matrix

    def draw_confusion_matrix(self):
        """Calculates the confusion matrix and shows it"""
        matrix = self.confusion_matrix(percentage=False, labeled=False)
        # TODO: add labels, title, legends, etc.
        pl.imshow(matrix, cmap="binary")
        pl.show()

    def get_n_samples(self, emotion, partition):
        """Returns number data samples of the `emotion` class in a particular `partition`
        ('test' or 'train')
        """
        if partition == "test":
            return len([y for y in self.y_test if y == emotion])
        elif partition == "train":
            return len([y for y in self.y_train if y == emotion])

    def get_samples_by_class(self):
        """
        Returns a dataframe that contains the number of training
        and testing samples for all emotions.
        Note that if data isn't loaded yet, it'll be loaded
        """
        if not self.data_loaded:
            self.load_data()
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(emotion, "train")
            n_test = self.get_n_samples(emotion, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)

        # get total
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total},
                            index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        """
        Returns random `emotion` data sample index on `partition`.
        """
        if partition == "train":
            index = random.choice(list(range(len(self.y_train))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_train))))
        elif partition == "test":
            index = random.choice(list(range(len(self.y_test))))
            while self.y_train[index] != emotion:
                index = random.choice(list(range(len(self.y_test))))
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index


def plot_histograms(classifiers=True, beta=0.5, n_classes=3, verbose=1):
    """
    Loads different estimators from `grid` folder and calculate some statistics to plot histograms.
    Params:
        classifiers (bool): if `True`, this will plot classifiers, regressors otherwise.
        beta (float): beta value for calculating fbeta score for various estimators.
        n_classes (int): number of classes
    """
    # get the estimators from the performed grid search result
    estimators = get_best_estimators(classifiers)

    final_result = {}
    for estimator, params, cv_score in estimators:
        final_result[estimator.__class__.__name__] = []
        for i in range(3):
            result = {}
            # initialize the class
            detector = EmotionRecognizer(estimator, verbose=0)
            # load the data
            detector.load_data()
            if i == 0:
                # first get 1% of sample data
                sample_size = 0.01
            elif i == 1:
                # second get 10% of sample data
                sample_size = 0.1
            elif i == 2:
                # last get all the data
                sample_size = 1
            # calculate number of training and testing samples
            n_train_samples = int(len(detector.X_train) * sample_size)
            n_test_samples = int(len(detector.X_test) * sample_size)
            # set the data
            detector.X_train = detector.X_train[:n_train_samples]
            detector.X_test = detector.X_test[:n_test_samples]
            detector.y_train = detector.y_train[:n_train_samples]
            detector.y_test = detector.y_test[:n_test_samples]
            # calculate train time
            t_train = time()
            detector.train()
            t_train = time() - t_train
            # calculate test time
            t_test = time()
            test_accuracy = detector.test_score()
            t_test = time() - t_test
            # set the result to the dictionary
            result['train_time'] = t_train
            result['pred_time'] = t_test
            result['acc_train'] = cv_score
            result['acc_test'] = test_accuracy
            result['f_train'] = detector.train_fbeta_score(beta)
            result['f_test'] = detector.test_fbeta_score(beta)
            if verbose:
                print(
                    f"[+] {estimator.__class__.__name__} with {sample_size * 100}% ({n_train_samples}) data samples achieved {cv_score * 100:.3f}% Validation Score in {t_train:.3f}s & {test_accuracy * 100:.3f}% Test Score in {t_test:.3f}s")
            # append the dictionary to the list of results
            final_result[estimator.__class__.__name__].append(result)
        if verbose:
            print()
    visualize(final_result, n_classes=n_classes)


def visualize(results, n_classes):
    """
    Visualization code to display results of various learners.

    inputs:
      - results: a dictionary of lists of dictionaries that contain various results on the corresponding estimator
      - n_classes: number of classes
    """

    n_estimators = len(results)

    # naive predictor
    accuracy = 1 / n_classes
    f1 = 1 / n_classes
    # Create figure
    fig, ax = pl.subplots(2, 4, figsize=(11, 7))
    # Constants
    bar_width = 0.4
    colors = [(random.random(), random.random(), random.random()) for _ in range(n_estimators)]
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                x = bar_width * n_estimators
                # Creative plot code
                ax[j // 3, j % 3].bar(i * x + k * (bar_width), results[learner][i][metric], width=bar_width,
                                      color=colors[k])
                ax[j // 3, j % 3].set_xticks([x - 0.2, x * 2 - 0.2, x * 3 - 0.2])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.2, x * 3))
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))
    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')
    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10)
    pl.tight_layout()
    pl.show()
