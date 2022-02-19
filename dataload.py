import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_diabetes_data(label_data_rate):

    diabetes_data = pd.read_csv("diabetes.csv")
    diabetes_data_copy = diabetes_data.copy(deep=True)

    diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
    diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
    diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
    diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
    diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)

    sc_X = StandardScaler()

    X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"], axis=1), ),
                     columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                              'BMI', 'DiabetesPedigreeFunction', 'Age'])
    y = diabetes_data_copy.Outcome

    X = np.asarray(X)
    y = np.asarray(pd.get_dummies(y))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5, random_state=42, stratify=y)

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))

    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]

    # Labeled data
    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx, :]

    return x_label, y_label, x_unlab, x_test, y_test