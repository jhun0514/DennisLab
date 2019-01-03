# Jong Hun Kim
# Pipeline example code for DennisLab
# Using digits datasets from sklearn module
# to test three different test methods and
# save best accuracy of them.

from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import warnings

# Load the data
digits = load_digits()
print('\nData and target shape of the digits dataset.')
print(digits.data.shape, digits.target.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

warnings.filterwarnings("ignore", category=FutureWarning)

# Construct three different pipelines
rfc_pipe = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=2)), ('method', ensemble.RandomForestClassifier(random_state=42))])

svm_pipe = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=2)), ('method', svm.SVC(random_state=42))])

abc_pipe = Pipeline([('ss', StandardScaler()), ('pca', PCA(n_components=2)), ('method', ensemble.AdaBoostClassifier(random_state=42))])

# Create a list of pipelines
pipelines = [rfc_pipe, svm_pipe, abc_pipe]

# Give a number to each methods
give_num = {0: 'Random Forest Classifier', 1: 'Support Vector Machine', 2: 'Ada Boost Classifier'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, y_train)

# Compare accuracies
for idx, val in enumerate(pipelines):
	print('\nPipeline test accuracy of %s: %.3f' % (give_num[idx], val.score(X_test, y_test)))

# Choose the best solution
accurancy = 0.0
methodB = 0
pipeB = ''
for idx, val in enumerate(pipelines):
	if val.score(X_test, y_test) > accurancy:
		accurancy = val.score(X_test, y_test)
		pipeB = val
		methodB = idx
print('\nBest accuracy classifier: %s' % give_num[methodB])

# Save the best accurancy pipeline to file
joblib.dump(pipeB, 'Pipeline.pkl', compress=1)
print('\nSaved %s pipeline to file' % give_num[methodB])
