# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:46:36 2024

@author: Kwangho Baek baek0040@umn.edu
"""
#%% Setup
import os
import pandas as pd
import numpy as np

import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable #, DefineVariable, bioMultSum
from biogeme.models import loglogit
from biogeme.database import Database


if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\NM-LCCM'
else:
    WPATH=os.path.abspath(r'C:\git\NM-LCCM')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)

dfMNL=pd.read_csv('dfMNL.csv')
dfMNL['tway']=0
dfMNL.loc[dfMNL.tiv>0,'tway']=1

if dfMNL.iv.min()<=0:
    raise Exception('remove iv<=0 paths')
dfMNL['alt'] = dfMNL.groupby('id').cumcount() + 1
maxalt=dfMNL.alt.max()

def ppForBiogeme(dfM,attrcols=['iv','ov','nTrans','PS','tway','wt','aux']):
    alts=[i+1 for i in range(maxalt)]
    comb = pd.MultiIndex.from_product([dfM['id'].unique(), alts], names=['id', 'alt'])
    dfLong = dfM.set_index(['id', 'alt']).reindex(comb, fill_value=0).reset_index()
    
    dfWide= dfLong.pivot(index='id', columns='alt',values=attrcols)
    dfWide.columns = [f'{col[0]}_{int(col[1])}' for col in dfWide.columns]
    dfWide=dfWide.reset_index()
    
    for alt in alts:
        dfWide[f'avail_{alt}'] = (dfWide[f'iv_{alt}'] != 0).astype(int)
    choice=dfLong.loc[dfLong.match==1,['id','alt']]
    choice.columns=['id','choice']
    dfWide = pd.merge(dfWide, choice, on='id')
    return dfWide

dfPP=ppForBiogeme(dfMNL)



database = Database('mymodel',dfPP)

# Define parameters (Betas) that will be estimated
BETA_IV = Beta('BETA_IV', 0, None, None, 0)  # Initial value is 0, no bounds
BETA_WT = Beta('BETA_WT', 0, None, None, 0)
BETA_AUX = Beta('BETA_AUX', 0, None, None, 0)
BETA_OV = Beta('BETA_OV', 0, None, None, 0)
BETA_NTRANS = Beta('BETA_NTRANS', 0, None,None, 0)
BETA_PS = Beta('BETA_PS', 0, None, None, 0)
BETA_TWAY = Beta('BETA_TWAY', 0, None, None, 0)


V = {}
av = {}

for i in range(1, maxalt + 1):
    # Define variables for each alternative dynamically
    iv = Variable(f'iv_{i}')
    ov = Variable(f'ov_{i}')
    wt = Variable(f'wt_{i}')
    aux = Variable(f'aux_{i}')
    nTrans = Variable(f'nTrans_{i}')
    PS = Variable(f'PS_{i}')
    tway=  Variable(f'tway_{i}')
    # Define utility function for alternative i
    #V[i] = BETA_IV * iv + BETA_WT * wt + BETA_AUX * aux + BETA_NTRANS * nTrans + BETA_PS * PS + BETA_TWAY * tway
    V[i] = BETA_IV * iv + BETA_OV * ov + BETA_NTRANS * nTrans + BETA_PS * PS + BETA_TWAY * tway
    # Define availability for alternative i (e.g., avail_1, avail_2, ..., avail_5)
    av[i] = Variable(f'avail_{i}')

choice = Variable('choice')



# Logit model: loglogit(V, av, chosen)
logprob = loglogit(V, av, choice)

# Define the likelihood function
biogeme_model = bio.BIOGEME(database, logprob)

# Estimate the model
results = biogeme_model.estimate()
print(results.get_estimated_parameters())


