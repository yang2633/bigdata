#### common  packages import
```python
import pandas as pd
import numpu as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
```

#### pandas:
```python 
import pandas as pd
# correlation heat map
corr = df.corr()
corr.style.background_gradient().set_precision(3)

# basic histgram for each dataframe columns
df.hist(bins, figsize=())

# find rows with NaN in df:
df_nulls = df[df.isnull().any(axis=1)]

# select rows:
df.loc[404, :]
df.iloc[404][:]

# remove rows with missing data:
df_nonulls = df.dropna()

# fill NaN with mean:
df.fillna(df.mean(), inplace=True)

# sort df by a column
df.sort_values(by=['column_name'], inplace=True)

```
#### sklearn:
train-test data set split:
```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                        shuffle = True,
                                         stratify = y,
                                    random_state = 440)
```
MinMaxScaler:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```
StandardScaler:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```
mean squared error:
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
```
shuffle pandas dataframe with sklearn
```python
from sklearn.utils import shuffle
df_new = shuffle(df)
# sometime resetting indexes is needed
df_new.reset_index(inplace=True, drop=True) 
# this will drop the original index
```
#### numpy:
#### concatenate
```python
np.concatenate([nparray1, nparray2....], axis=1)
```
#### useful functions:
one hot encoding with Sklaern.preprocessing.FunctionTransformer
```python
def one_hot(df):
	df_copy = df.copy()
	one_hot = pd.get_dummies(df[feature]) # feature is the one needs  to be encoded
	new_feats = [str(x) for x in list(one_hot.columns)]
    df_copy[new_feats[:-1]] = one_hot[one_hot.columns[:-1]]
    keep = list(df_copy.columns)
    keep.remove(str(feature))
    return df_copy[keep]
```
custom scaler
```python 
from sklearn.base BaseEstimator, TransformerMixin
class CustomScaler(BaseEstimator, TrandformerMinin):
	def __init__(self):
		self.scaler = StandardScaler()
		self.feats = [feats]
	def fit(self, X, y=None):
		self.scaler.fit(X[self.feats])
		return self
	def transform(self, X, y=None):
		X_copy = X.copy()
		X_copy[self.feats] = self.scaler.transform(X_copy[self.feats])
		return X_copy

# in a pipe: ...("scaler", CustomScaler)...
```
nested_defaultdict (useful for confusionMatrix)
```python
from collections import defaultdict
from functools import partial
from itertools import repeat
def nested_defaultdict(default_factory, depth=1):
	result = partial(defaultdict, default_factory)
	for _ in repeat(None, depth-1):
		result = partial(defaultdict, result)
	return result()
# intilize with counter = nested_defaultdict(int, 2)
```
plot a image (mnist number)
```python
# with matplotlib
n = row_number
row = df.iloc(n)
img = row[:784]
img = img.values.reshape(28,28)
plt.imshow(img)

# with plotly
import chart_studio.plotly as py
import plotly.graph_objs as o
l1 = []
l2 = []
for i in range(28):
	l1.append(i)
	l2.append(27-i)
row = df.iloc[666]
img = row[:784].values.reshape(28,28)
trace = go.Heatmap(
	z = img,
	x = np.array(l1),
	y = np.array(l2),
	colorscale = 'Viridis'
)
data = [trace]
layout = go.Layout(
	autosize = False,
	width=500,
	height=500
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```
Classication performance measure
confusion matrix, precission, recall, f1 score, ROC, and AUC
```python
from skleaen import metrics
def binaryPerformance(y, y_pred, y_score):
	cm = nest_defaultdict(int, 2)
	for i in range(len(y_pred)):
		trueClass = y[i]
		predClass = y_pred[i]
		cm[trueClass][predClass] += 1
	#print(cm)
	tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2.0/((1.0/precision)+(1.0/recall))
    data = [[cm[1][1],cm[1][0]],[cm[0][1],cm[0][0]]]
    df_temp = pd.DataFrame(data)
    df_temp.rename(columns={0: "pred=target", 1:"pred=nottarget"}, 
                   index={0:'True=target', 1:'True=notTarget'},
                   inplace=True)
    #print("confusionMatrix")
    #print(df_temp)
    fpr, tpr, threshold = metrics.roc_curve(y, y_pred, pos_label=None)
    auc = metrics.roc_auc_score(y, y_pred)
    return precision, recall, auc, fpr, tpr, thresholds
```
manually extract preformance
```python
# given y and y_pred
# cm
cm = nexted_defaultdict(int, 2)
for i in range(len(y_pred)):
	trueClass = y[i]
	predClass = y_pred[i]
	cm[trueClass][predClass] += 1
# make a dataframe from cm
data = [[cm[A][A], cm[A][B]],
		[cm[B][A], cm[B][B]]]
cm_df = pd.DataFrame(data)
cm_df.rename(columns={0:"pred=A", 1: "pred=B"}, 
          index={0:'true=A', 1:'true=B'},
         inplace=True)
# tp, fp, tn, fn
tp = cm[digitA][digitA]
fp = cm[digitB][digitA]
fn = cm[digitA][digitB]
tn = cm[digitB][digitB]
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2.0/((1.0/precision)+(1.0/recall))

# get ROC from estimator's scores
y_scores = clf.decision_function(X_test)
signal_scores = []
background_scores = []
for i in range(len(y_test_pred)):
    if y_test[i] == digitA:
        signal_scores.append(y_scores[i])
    else:
        background_scores.append(y_scores[i])
maxScore = max(max(signal_scores), max(background_scores))
minScore = min(min(signal_scores), min(background_scores))
trace0 = go.Histogram(
    x = signal_scores,
    name = 'Signal',
    opacity=0.6
)
trace1 = go.Histogram(
    x = background_scores,
    name = 'Background',
    opacity=0.6
)
data = [trace0, trace1]
layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='overlaid histogram')

length_signal = float(len(background_scores))
length_background = float(len(signal_scores))

stepSize = (maxScore-minScore) / 100.0
threshold = minScore
fprs = []
tprs = []
thresholds = []

for steps in range(100):
    tn = 0.0
    tp = 0.0
    for score in background_scores:
        if score <= threshold:
            tn += 1.0
    for score in signal_scores:
        if score > threshold:
            tp += 1.0
    fn = length_signal- tp
    fp = length_background - tn
    tpr = tp/length_signal
    fpr = fp/length_background
    tprs.append(tpr)
    fprs.append(fpr)
    thresholds.append("Threshold="+str(threshold))
    threshold += stepSize
    
trace0 = go.Scatter(
    x = fprs,
    y = tprs,
    text = thresholds,
    mode='markers'
)
layout = dict(
    title = "ROC Curve",
    xaxis = dict(title='FPR'),
    yaxis = dict(title='TPR')
)

data = [trace0]
iplot(dict(data=data, layout=layout))
```
plot ROC with plotly
```python
import chart_studio.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go

trace0 = go.Scatter(
	x = fprs,
	y = tprs,
	text = thresholds,
	mode = 'lines',
	name = "training"
)
layout = dict(
	title="ROC curve"
	xaxis = dict(title="FPR")
	yaxis = dict(title="TPR")
)
data = [trace0]
iplot(dict(data=data, layout=layout))
```