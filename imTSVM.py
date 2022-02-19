import numpy as np
import sklearn.svm as svm
from dataload import load_diabetes_data
from utils import matrix2vec, eval

class ITSVM():

    def __init__(self, kernel='linear', Np=50, C=1.0, a=0.01 ):

        self.Np = Np
        self.C = C
        self.kernel = kernel
        self.clf = svm.SVC(C=1.5, kernel=self.kernel, probability=True)
        self.a = a

    def train(self, X1, Y1, X2):

        if len(Y1.shape) > 1:
            Y1 = matrix2vec(Y1)

        N = len(X1) + len(X2)

        Y1[Y1 == 0] = -1

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)

        self.Nn = len(X2)-self.Np
        self.C_p = self.a * self.Nn
        self.C_n = self.a * self.Np

        X3 = np.r_[X1, X2]

        while (self.C_p < self.C) or (self.C_n < self.C):
            sample_weights = np.r_[Y1, Y2].copy()
            for i, item in enumerate(sample_weights):
                if i < len(X1):
                    sample_weights[i] = 1.
                elif item == 1:
                    sample_weights[i] = self.C_p
                elif item == -1:
                    sample_weights[i] = self.C_n

            self.clf.fit(X3, np.r_[Y1, Y2], sample_weight=sample_weights)

            dist = self.clf.decision_function(X2)
            e = 1-dist*Y2

            pos_index = np.where(Y2 == 1)[0]
            neg_index = np.where(Y2 == -1)[0]

            delta = e[pos_index].sum() - e[neg_index].sum()

            pos_cons_index = []
            neg_cons_index = []

            for i, item in enumerate(e[pos_index]):
                if item > delta/(self.Nn+1) + (self.Np-1)/(self.Nn+1) * max((2-item), 0):
                    pos_cons_index.append(pos_index[i])

            for i, item in enumerate(e[neg_index]):
                if item > -delta/(self.Np+1) + (self.Nn-1)/(self.Np+1) * max((2-item), 0):
                    neg_cons_index.append(neg_index[i])

            if len(pos_cons_index) != 0 or len(neg_cons_index) != 0:

                if len(pos_cons_index) != 0:
                    max_pos_e_index = pos_cons_index[np.argmax(e[pos_cons_index])]
                    Y2[max_pos_e_index] = -Y2[max_pos_e_index]
                    self.Np = self.Np - 1
                    self.Nn = self.Nn + 1
                    self.C_p = self.a * self.Nn
                    self.C_n = self.a * self.Np

                if len(neg_cons_index) != 0:
                    max_neg_e_index = neg_cons_index[np.argmax(e[neg_cons_index])]
                    Y2[max_neg_e_index] = -Y2[max_neg_e_index]
                    self.Np = self.Np + 1
                    self.Nn = self.Nn - 1
                    self.C_p = self.a * self.Nn
                    self.C_n = self.a * self.Np

            else:
                self.C_p = min(2*self.C_p, self.C)
                self.C_n = min(2*self.C_n, self.C)


    def predict(self, x):

        return self.clf.predict(x)

    def predict_proba(self, x):

        return self.clf.predict_proba(x)

if __name__ == '__main__':

    x_label, y_label, x_unlab, x_test, y_test = load_diabetes_data(0.1)

    itsvm = ITSVM()
    itsvm.train(x_label, y_label, x_unlab)

    y_pre = itsvm.predict_proba(x_test)

    acc = eval('acc', y_test, y_pre)

    print('accuracy:', acc)






