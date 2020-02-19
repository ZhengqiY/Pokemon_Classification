import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# load dataset
train_x = np.load("pokemon_train_x.npy")
train_y = np.load("pokemon_train_y.npy")

# Preprocessing Training dataset X
x_processed = []

for t in train_x:
    list_x = list(t)
    if list_x[9] == 'F':
        list_x[9] = 0
    else:
        list_x[9] = 1
    x_processed.append(list_x)

# Preprocessing Training label set Y
np.unique(train_y)
y_processed = []

for names in train_y:
    if names == 'Bulbasaur':
        names = 2
    elif names == 'Charmander':
        names = 3
    elif names == 'Gastly':
        names = 7
    elif names == 'Jigglypuff':
        names = 10
    elif names == 'Pidgey':
        names = 9
    elif names == 'Pikachu':
        names = 11
    elif names == 'Squirtle':
        names = 19
    elif names == 'Sudowoodo':
        names = 21
    y_processed.append(names)

# split training set (80%) and validation set (20%)
x_train, x_vali, y_train, y_true = train_test_split(x_processed,  \
                                y_processed, test_size = 0.2)


# train KNN model with training dataset with k = 80 = sqrt(8000*0.8)
neigh = KNeighborsClassifier(n_neighbors=80)
neigh.fit(x_train, y_train)
KNeighborsClassifier(...)
y_pred = neigh.predict(x_vali)
# confusion matrix
confusion_matrix(y_true, y_pred)
# classification result
target_names = list(np.unique(train_y))
print(classification_report(y_true, y_pred, target_names = target_names))

# Normalize the training set
min_max_scaler = preprocessing.MinMaxScaler()
x_norm = min_max_scaler.fit_transform(x_train)
x_vali_norm = min_max_scaler.fit_transform(x_vali)

# Train and evaluate SVM Model
clf_svm = SVC(gamma = 'auto')
clf_svm.fit(x_norm, y_train)
y_pred_svm = clf_svm.predict(x_vali_norm)
confusion_matrix(y_true, y_pred_svm)
print(classification_report(y_true, y_pred_svm, target_names = target_names))

# Train and evaluate DT Model
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(x_train, y_train)
y_pred_tree = clf_tree.predict(x_vali)
confusion_matrix(y_true, y_pred_tree)
print(classification_report(y_true, y_pred_tree, target_names = target_names))

# Train and evaluate LDA Model
clf_lda = LinearDiscriminantAnalysis()
clf_lda = clf_lda.fit(x_train, y_train)
y_pred_lda = clf_lda.predict(x_vali)
confusion_matrix(y_true, y_pred_lda)
print(classification_report(y_true, y_pred_lda, target_names = target_names))

# Train and evaluate NB Model
clf_nb = GaussianNB()
clf_nb = clf_nb.fit(x_train, y_train)
y_pred_nb = clf_nb.predict(x_vali)
confusion_matrix(y_true, y_pred_nb)
print(classification_report(y_true, y_pred_nb, target_names = target_names))

# Train and evaluate NN Model
clf_mlp = MLPClassifier(solver = 'adam', alpha = 1e-5,  \
                    hidden_layer_sizes = (32,16,8), \
                       activation = 'relu')
clf_mlp = clf_mlp.fit(x_train, y_train)
y_pred_mlp = clf_mlp.predict(x_vali)
confusion_matrix(y_true, y_pred_mlp)
print(classification_report(y_true, y_pred_mlp, target_names = target_names))

# load testing dataset
test = np.load("pokemon_test_x.npy")
# preprocess testing dataset
test_processed = []
for t in test:
    list_t = list(t)
    if list_t[9] == 'F':
        list_t[9] = 0
    else:
        list_t[9] = 1
    test_processed.append(list_t)
test_pred = clf_tree.predict(test_processed)