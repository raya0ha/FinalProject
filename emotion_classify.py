import pickle
import warnings
warnings.filterwarnings('ignore')
from utils import extract_feature


class emotion_model(object):
    def __init__(self):
        self.model = pickle.load(open("mlp_classifier.model", "rb"))

    def predict(self, filename):
        # extract features and reshape it
        features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        # predict
        result = self.model.predict(features)[0]
        # show the result !
        print(filename, result[0])
        return result

#predict file
print(emotion_model().predict('C:\\Users\\windows\\PycharmProjects\\RayaWesal\\data\\Actor_03\\03-01-01-01-01-01-03.wav'))
print(emotion_model().predict('C:\\Users\\windows\\PycharmProjects\\RayaWesal\\data\\Actor_19\\03-01-08-02-02-02-19.wav'))
