

import streamlit as st
st.title("SELECT CLASSIFIER FOR YOUR DATASET:")
st.write("""
##### Explore different classifiers""")
Dataset_name= st.sidebar.selectbox("Select Dataset: ", ("Iris","Breast Cancer", "Wine dataset"))
st.write(Dataset_name)

classifier_name = st.sidebar.selectbox("Select classifier: ", ("KNN","SVM", "Random Forest"))

import numpy as np
from sklearn import datasets
def get_dataset(Dataset_name):
    if Dataset_name=="Iris":
        data= datasets.load_iris()
    elif Dataset_name =="Breast Cancer":
        data= datasets.load_breast_cancer()
    else:
        data= datasets.load_wine()
    X= data.data
    Y= data.target
    return X,Y
X,Y= get_dataset(Dataset_name)
st.write("Shape of Dataset:", X.shape)
st.write("Number of classes:", len(np.unique(Y)))

def add_params_ui(clf_name):
    params= dict()
    if clf_name=="KNN":
        K= st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name=="SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth",2,15)
        n_estimators = st.sidebar.slider("No.of trees",1,100)
        params["max_depth"]= max_depth
        params["n_estimators"]  = n_estimators
    return params
params= add_params_ui(classifier_name)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf= KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth= params["max_depth"], random_state=1234)
    return clf
clf = get_classifier(classifier_name,params)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=1234)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,y_pred)
st.write(f"Classifier :{classifier_name}")
st.write(f"Accuracy: {accuracy}")

#Plotting
from sklearn.decomposition import PCA
pca= PCA(2)
X_projected = pca.fit_transform(X)

import matplotlib.pyplot as plt
x1 = X_projected[:,0]
x2=X_projected[:,1]
fig = plt.figure()
plt.scatter(x1,x2,c=Y, alpha=0.8,cmap = "viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)










