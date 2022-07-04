# End-to-end ML Model Creation [![gcash donation][1]][2] [![paypal donation][3]][4]

[![python version][7]][8] [![scikit version][11]][12] [![build][13]][14] 
 
This repository is a collection of notebooks that analyzes thousand datasets on heart-disease in order to try and predict 
if a user has heart disease. The dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). 
![image](https://user-images.githubusercontent.com/102983286/177067798-62235264-6cf7-40e9-a3bf-22c02017d2e5.png) 


## 1. Problem Definition
_What problem are we trying to solve?_

This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

The objective of this model creation is to classify if our incoming patients have heart disease based on the same set of tests.

* __Type of Machine Learning__: _Supervised Learning_
* __Type of Learning__: _Classification_

## 2. Data
Our main dataset for this project  was downloaded from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). It is very important to have knowledge on what our data looks like and what it represents. Furthermore, it would also be important for our analysis that we get to categorize the different variables of our data so we can check the data accuracy. We will also be showing how to filter and sort our data in this section.

* __Type of Data __
    * Based on Structure: 
        * _Structured_
        * Unstructured
    * Based on Frequency: 
        * _Static_
        * Streaming
        
 ## 3. Success Criteria
_What defines accuracy?_
We calculate accuracy by dividing the number of correct predictions (the corresponding diagonal in the matrix) by the total number of samples.
![image](https://user-images.githubusercontent.com/102983286/177068427-ec615a73-49af-438a-85e5-8eb19db5fad4.png)
 ```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_actual    = [1,1,1,1,0,0,0,0,0,0]
y_predicted = [1,1,1,0,1,1,1,1,0,0]

cm = confusion_matrix(y_actual, y_predicted)
cmd = ConfusionMatrixDisplay(cm)
cmd.plot();
```
![image](https://user-images.githubusercontent.com/102983286/177068539-8bdd43d3-1cb7-491e-9810-29e9acb182c9.png)

## 4. Features
_What features does our data have and which ones to use?_
The features that we have are:
1. Binary
    * sex  (Male or Female)
        * 0 = female 
        * 1 = male
    * fbs (Fasting Blood Sugar > 120 mg/dl)
        * 0 = no
        * 1 = yes
    * exang (Exercise Induced Angina)
        * 0 = no
        * 1 = yes
    * target (Heart Disease / Target Field)
        * 0 = disease
        * 1 = no disease
        
2. Categorical
    * cp (Chest Pain Type)
        * 0: asymptomatic
        * 1: atypical angina
        * 2: non-anginal pain
        * 3: typical angina
    * restecg (Resting ECG)
        * 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
        * 1: normal
        * 2: having ST-T wave abnormality
    * slope (the slope of the peak exercise ST segment)
        * 0: downsloping
        * 1: flat
        * 2: upsloping
    * ca (number of major vessels)
        * (0–3)
    * thal (Thalassemia)
        * 1 = normal
        * 2 = fixed defect
        * 3 = reversible defect
3. Continuous
    * age (Age of the individual)
    * trestbps (Resting Blood Pressure in mm/hg) 
    * chol (Serum Cholesterol in mg/dl)
    * thalac (Maximum heart rate achieved)
    * oldpeak (ST depression induced by exercise relative to rest)

## 5. Modelling
_What kind of model should we use? How to use a model?_

Finding the best estimator for the task might frequently be the most difficult step in tackling a machine learning challenge. For various data kinds and issues, other estimators are more appropriate.
![title](https://scikit-learn.org/stable/_static/ml_map.png)

### 5.1 LinearSVC
The Linear SVM produced an accuracy of 79.51% and we can optimize our features or modify hyperparamters but let us try and look at the sci-kit learn estimator chart.
![image](https://user-images.githubusercontent.com/102983286/177069227-caf6bd37-f130-4b85-bdde-5c4a4afe275a.png)


### 5.2 Naive Bayes
Our naive-bayes classification has produced a good accuracy of 80% already but having 20 incorrect heart disease prediction out of 100 patients seems a little bit high, let us try implementing another model.
![image](https://user-images.githubusercontent.com/102983286/177069042-700af1de-434e-4185-ad3a-3a0df30d38c6.png)

#### 5.3 Random Forest
Our random forest model, produced 92.98% accuracy. Here is the last tree from our random forest.
![image](https://user-images.githubusercontent.com/102983286/177069323-eede9b6f-10bf-4812-be92-d422aaa1430d.png)

## 6. Deployment
_How can we share our model?_
```python
import pickle
pickle.dump(clf, open("heart_disease_random_forest_model.pkl", "wb"))
```
We can export our models into a pickle file and from there, deploy it on the web. For deployment, you may access the [heart-disease repository](https://github.com/mcabanlit/heart-disease) which was deployed to [Heroku](https://heart-disease-pywebio.herokuapp.com/). 

[1]: https://img.shields.io/badge/donate-gcash-green
[2]: https://drive.google.com/file/d/1JeMx5_S7VBBT-3xO7mV9YOMfESeV3eKa/view

[3]: https://img.shields.io/badge/donate-paypal-blue
[4]: https://www.paypal.com/paypalme/mcabanlitph

[5]: https://img.shields.io/badge/license-GNUGPLv3-blue.svg
[6]: https://github.com/mcabanlit/heart-disease/blob/main/LICENSE.md

[7]: https://img.shields.io/badge/python-3.10-blue
[8]: https://www.python.org/

[9]: https://img.shields.io/badge/pywebio-1.6.1-dark
[10]: https://pywebio.readthedocs.io/en/latest/

[11]: https://img.shields.io/badge/scikit--learn-1.1.1-orange
[12]: https://scikit-learn.org

[13]: https://img.shields.io/badge/build-passing-green
[14]: https://heart-disease-pywebio.herokuapp.com/
