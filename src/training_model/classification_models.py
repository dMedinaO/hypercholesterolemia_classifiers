import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (precision_score, 
                             recall_score, 
                             f1_score, 
                             accuracy_score, 
                             matthews_corrcoef,
                             confusion_matrix)

from sklearn.linear_model import (RidgeClassifier, 
                                  LogisticRegression,
                                  SGDClassifier)

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

from sklearn.svm import (LinearSVC, NuSVC, SVC)
from sklearn.neighbors import (KNeighborsClassifier, RadiusNeighborsClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, GaussianNB)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

class ClassificationModel(object):

    def __init__(self, 
                 train_values=None, 
                 test_values=None, 
                 train_response=None, 
                 test_response=None) -> None:
        
        self.train_values = train_values
        self.test_values = test_values
        self.train_response = train_response
        self.test_response = test_response

        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

    def __get_metrics(self, y_true=None, y_predict=None):

        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_predict).ravel()

        sensitivity = tp/(tp + fn)
        specificity = tn/(fp + tn)

        row = [
            accuracy_score(y_true=y_true, y_pred=y_predict),
            precision_score(y_true=y_true, y_pred=y_predict),
            recall_score(y_true=y_true, y_pred=y_predict),
            f1_score(y_true=y_true, y_pred=y_predict),
            matthews_corrcoef(y_true=y_true, y_pred=y_predict),
            sensitivity,
            specificity
        ]

        return row

    def __process_performance_cross_val(self, performances):
        
        row_response = []
        for i in range(len(self.keys)):
            value = np.mean(performances[self.keys[i]])
            row_response.append(value)
        return row_response
    
    def __apply_model(self, model=None, description=None):

        model.fit(self.train_values, self.train_response)
        predictions = model.predict(self.test_values)

        metrics_validation = self.__get_metrics(
            y_true=self.test_response,
            y_predict=predictions)
        
        response_cv = cross_validate(
            model, 
            self.train_values, 
            self.train_response, 
            cv=5, 
            scoring=self.scores)

        metrics_cv = self.__process_performance_cross_val(
            response_cv
        )

        row = [description] + metrics_cv + metrics_validation

        return row
    
    def apply_exploring(self):

        matrix_response = []

        try:
            clf_model = RidgeClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Ridge"
                )
            )
        except:
            pass

        try:
            clf_model = LogisticRegression(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Logistic"
                )
            )
        except:
            pass
        
        try:
            clf_model = SGDClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="SGD"
                )
            )
        except:
            pass
        
        try:
            clf_model = LinearDiscriminantAnalysis()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LDA"
                )
            )
        except:
            pass
        
        try:
            clf_model = QuadraticDiscriminantAnalysis()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="QDA"
                )
            )
        except:
            pass
        
        try:
            clf_model = LinearSVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LinearSVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = SVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="SVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = NuSVC(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="NuSVC"
                )
            )
        except:
            pass
        
        try:
            clf_model = KNeighborsClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="KNN"
                )
            )
        except:
            pass
        
        try:
            clf_model = RadiusNeighborsClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="RNN"
                )
            )
        except:
            pass
        
        try:
            clf_model = GaussianProcessClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GPC"
                )
            )
        except:
            pass
        
        try:
            clf_model = BernoulliNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="BernoulliNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = CategoricalNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="CategoricalNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = GaussianNB()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GaussianNB"
                )
            )
        except:
            pass
        
        try:
            clf_model = DecisionTreeClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="DecisionTree"
                )
            )
        except:
            pass
        
        try:
            clf_model = ExtraTreeClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="ExtraTree"
                )
            )
        except:
            pass
        
        try:
            clf_model = RandomForestClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="RandomForest"
                )
            )
        except:
            pass

        try:
            clf_model = BaggingClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="Bagging"
                )
            )
        except:
            pass
        
        try:
            clf_model = AdaBoostClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="AdaBoost"
                )
            )
        except:
            pass
        
        try:
            clf_model = GradientBoostingClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="GradientBoosting"
                )
            )
        except:
            pass
        
        try:
            clf_model = HistGradientBoostingClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="HistGradientBoosting"
                )
            )
        except:
            pass
        
        try:
            clf_model = ExtraTreesClassifier(
                
            )
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="ExtraTrees-ensemble"
                )
            )
        except:
            pass
        
        try:
            clf_model = XGBClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="XGB"
                )
            )
        except:
            pass
        
        try:
            clf_model = LGBMClassifier()
            matrix_response.append(
                self.__apply_model(
                    model=clf_model,
                    description="LGBM"
                )
            )
        except:
            pass

        header = ["algorithm", 'fit_time', 'score_time', 'F1_cv', 'recall_cv', 'precision_cv', 'accuracy_cv', 
                  'accuracy_val', 'precision_val', 'recall_val', 'f1_val', 'matthews_corrcoef_val', 'sensitivity', 'specificity']

        df_summary = pd.DataFrame(data=matrix_response, 
                                  columns=header)
        
        return df_summary