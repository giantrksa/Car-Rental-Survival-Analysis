<!-- #region -->
# Survival Analysis Project

Welcome to our project! Here, we're going to do some Survival Analysis with some nifty Python libraries. The objective of this project is to illustrate the survival probabilities of a given population.

### Libraries and Data
To start off, we use libraries such as Pandas, NumPy, Seaborn, Matplotlib, and scikit-survival. Our initial dataset is taken from the `load_veterans_lung_cancer()` function from the `sksurv.datasets` package.

```python
import pandas as pd
import numpy as np
from sksurv.datasets import load_veterans_lung_cancer
data_x, data_y = load_veterans_lung_cancer()
```
In this dataset, each row represents a veteran diagnosed with lung cancer. The `Survival_in_days` field represents the number of days the veteran survived after the diagnosis.

### Exploratory Data Analysis
We first look at the data using the Kaplan Meier estimator and plot it.

```python
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt

time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
```
Then we check the distribution of the survival days using Seaborn's `displot()` function.

```python
import seaborn as sns
df_y = pd.DataFrame(data_y)
sns.displot(df_y['Survival_in_days'])
```
### Dataset Generation
Next, we generate a new dataset which includes 'Date', 'Distance', and 'Service Period' fields.

```python
from datetime import datetime
import random

series = pd.date_range(start='2021-01-01', end=datetime.now(), freq='D')
distance = np.sort(random.sample(range(50000), len(series)))
binary = [True, False]
service = np.random.choice(binary, len(distance), p=[0.2, 0.8])

df = pd.DataFrame({'Date':series, 'Distance':distance, 'Service Period':service})
```

### Modeling
For the new dataset, we calculate survival probabilities based on Service Period. We generate two types of plots: one considering time and another considering distance.

```python
time, survival_prob = kaplan_meier_estimator(df["Service Period"], df["Date"].index)
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")

dist, survival_prob = kaplan_meier_estimator(df["Service Period"], df["Distance"])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("Distance ($km$)")
```
We then create a dataframe that includes Day, Distance, and Survival Probability fields.

```python
dff_surv = pd.DataFrame({'Day': time, 'Distance':dist, 'Survival Probability':survival_prob})
```
Using Plotly, we create a survival analysis plot with color-coded regions representing levels of survival probability.

```python
import plotly.express as px

fig = px.line(dff_surv, x="Day", y="Survival Probability", hover_data = ['Distance'],title='Car Rental Survival Analysis', )
fig.add_hrect(y0=0.95, y1=1.05, line_width=0, fillcolor="green", opacity=0.2)
fig.add_hrect(y0=0.80, y1=

0.95, line_width=0, fillcolor="orange", opacity=0.2)
fig.add_hrect(y0=min(survival_prob) - 0.05, y1=0.80, line_width=0, fillcolor="red", opacity=0.2)
fig.add_vline(x=dff_surv[dff_surv['Survival Probability'] >= 0.95][-1:]['Day'].values[0], line_width=3, line_dash="dash", line_color="green")
fig.add_vline(x=dff_surv[dff_surv['Survival Probability'] >= 0.80][-1:]['Day'].values[0], line_width=3, line_dash="dash", line_color="orange")

fig.show()
```
Finally, we built a Python class `graph_survival` to encapsulate the code for generating the survival analysis graph, so it can be easily reused.

```python
class graph_survival():
    '''
    software Car Performance Indicator
    @author : Gian Antariksa (CBI)
    '''
    def __init__(self, df, X_col, y_col, hover_col, color_list, limit_list, title_graph='Car Rental Survival Analysis'):
        # Code omitted for brevity. Please refer to original code.

    def plain_graph(self):
        # Code omitted for brevity. Please refer to original code.

    def complete_graph(self):
        # Code omitted for brevity. Please refer to original code.
```
This class includes functions to generate both plain and complete graphs. We can use it like this:

```python
color_list=['green','orange','red']
limit_list = [0.95, 0.8]
graph_survival(dff_surv, 'Day','Survival Probability', 'Distance', color_list, limit_list).plain_graph()
```
And that's the walkthrough of our code! We hope you enjoy the journey of survival analysis with us. Cheers!
<!-- #endregion -->

```python

```
