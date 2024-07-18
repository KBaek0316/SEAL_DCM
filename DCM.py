# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:46:36 2024

@author: Kwangho Baek baek0040@umn.edu
"""
#%% Setup
import os
import pandas as pd
import numpy as np
from biogeme import models, biogeme
from biogeme.expressions import Beta, Variable,PanelLikelihoodTrajectory
from biogeme.database import Database

if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\NM-LCCM'
else:
    WPATH=os.path.abspath(r'C:\git\NM-LCCM')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)

dfMNL=pd.read_csv('dfMNL.csv')
dfWide=dfMNL[dfMNL.tway==1].copy().rename(columns={"aux": "aux_t","wt": "wt_t","iv": "iv_t","nTrans": "nTrans_t"})
dfWide['alt']=2
dfWide.loc[dfWide.match==1,'alt']=1
dfSub=dfMNL[dfMNL.tway==0].copy().rename(columns={"aux": "aux_n","wt": "wt_n","iv": "iv_n","nTrans": "nTrans_n"})


dfWide=pd.merge(dfWide,dfSub[['id','aux_n','wt_n','iv_n','nTrans_n']],how='left',on='id')



#database = Database('database', dfMNL)
database = Database('database', dfWide)

# Define individual-specific variables
#Income = Variable('Income')
#Age = Variable('Age')

# Define alternative-specific variables
aux_t = Variable('aux_t')
wt_t = Variable('wt_t')
iv_t = Variable('iv_t')
nTrans_t = Variable('nTrans_t')

aux_n = Variable('aux_n')
wt_n = Variable('wt_n')
iv_n = Variable('iv_n')
nTrans_n = Variable('nTrans_n')


# Define choice variable
Choice = Variable('match')

# Define the alternative identifier
Alternative = Variable('alt')

# Define the parameters to be estimated
ASC = Beta('ASC', 0, None, None, 0)
B_aux = Beta('B_AUX', 0, None, 0, 0)
B_wt = Beta('B_WT', 0, None, 0, 0)
B_iv = Beta('B_IV', 0, None, 0, 0)
B_nTrans = Beta('B_NTRANS', 0, None, 0, 0)

# Utility functions
V1=ASC+B_aux*aux_t+B_wt*wt_t+B_iv*iv_t+B_nTrans*nTrans_t
V2=B_aux*aux_n+B_wt*wt_n+B_iv*iv_n+B_nTrans*nTrans_n

# Associate utility functions with alternatives
V = {1: V1, 2: V2}

# Assuming all alternatives are available for all individuals
av = {1: 1, 2: 1}

# Define the model (logit model in this case)
logit_model = models.logit(V, av, Choice)
logprob = PanelLikelihoodTrajectory(logit_model)

biogeme_object = biogeme.BIOGEME(database, logit_model,logprob)

# Estimate the model
results = biogeme_object.estimate()

# Print the results
print(results.getEstimatedParameters())

