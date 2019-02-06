
# coding: utf-8

# In[418]:


import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import preprocessing
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, lars_path, Lasso, enet_path


# In[419]:


dataset = "Documents\\CellDNA1.csv"
DNA = pd.read_csv( dataset,header=None, )
dataset = np.genfromtxt("Documents\\CellDNA1.csv", delimiter = ',')



# In[420]:


DNA


# In[421]:


from scipy.stats import zscore
nz = DNA.apply(zscore)


# In[422]:


nz


# In[423]:


p = DNA.loc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]


# In[424]:


p


# In[425]:


from scipy.stats import zscore
n1 = p.apply(zscore)


# In[426]:


n1


# In[427]:


y1 = DNA.loc[:, [13]]


# In[428]:


y1


# In[429]:


y1.loc[y1[13] > 0, [13]] = 1


# In[430]:


y1


# In[431]:


# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Pickle package
import pickle
from matplotlib import style
style.use("ggplot")

import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[432]:


nn


# In[433]:


X=n1


# In[434]:


X


# In[435]:


y = y1


# In[471]:




clf = svm.SVC(kernel = 'linear', C = 1000, probability = True) #
clf.fit(X, y)



# In[563]:


clf.support_vectors_.shape


# In[472]:


print(clf.n_support_) # number of support vectors in EACH class


# In[496]:


print(clf.decision_function(X), '\n') 


# In[533]:


L = clf.decision_function(X)# w^tX+b


# In[534]:


L


# In[536]:


p = np.absolute(L) # absolute after 


# In[537]:


p


# In[540]:


k = np.sort(p) # absolute least to great


# In[541]:


k


# In[562]:


k_features = k.tolist()
k_features


# In[550]:


k.shape


# In[551]:


np.argsort(p) # indexes of absolute value least


# In[555]:


p[131]


# In[556]:


p[165]


# In[557]:


p[892]


# In[558]:


p[1057]


# In[459]:


print(clf.predict_proba(X), '\n')


# In[480]:


clf.decision_function(X).shape


# In[437]:



#print(clf.coef_) # coefficients in ”primary” form
#print(clf.dual_coef_) # coefficients in ”dual” form
a = clf.coef_
b= clf.dual_coef_


# In[ ]:






# In[ ]:





# In[ ]:




