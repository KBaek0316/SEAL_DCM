# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:50 2024

@author: Kwangho Baek baek0040@umn.edu
"""
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\NM-LCCM'
else:
    WPATH=os.path.abspath(r'C:\git\NM-LCCM')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)

random.seed(5723588)
torch.manual_seed(5723588)
np.random.seed(5723588)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
TwayDict={'METRO Blue':'901','METRO Green':'902','METRO Red':'903',
          'METRO Orange':'904','METRO A':'921','METRO C':'923'}

def InputProcessing(surFile,pathFile,ver,convFile): #implemented in each session
    '''for debugging
    surFile='survey'
    pathFile='paths'
    ver=2022
    convFile='dfConv'
    '''
    dfSurveyRaw=pd.read_csv(surFile+str(ver)+r'.csv',low_memory=False, encoding='ISO 8859-1')
    dfConversion=pd.read_csv(convFile+r'.csv')
    #VER=int(pd.to_datetime(dfSurveyRaw.survey_date).dt.year.median())
    match ver:
        case 2016:
            dfSurvey=dfSurveyRaw.loc[:,['ID','DATE','ORIGIN_LAT_100M','ORIGIN_LON_100M',
                                     'TIME_ON','ACCESS_MODE',
                                     'TRANSFERS_FROM_FIRST','TRANSFERS_FROM_SECOND','TRANSFERS_FROM_THIRD','ROUTE_SURVEYED',
                                     'TRANSFERS_TO_FIRST','TRANSFERS_TO_SECOND','TRANSFERS_TO_THIRD',
                                     'EGRESS_MODE','DESTINATION_LAT_100M','DESTINATION_LON_100M']]
            dfSurvey['acwait']=np.nan
            dfSurvey['DATE']=pd.to_datetime(dfSurvey.DATE,format="%A, %B %d, %Y")
            routeConv={'BLUE METRO':'901','GREEN METRO':'902','RED METRO':'903',
                       'ALINE Northbound':'921','ALINE Southbound':'921', 'Other':np.nan}
        case 2022:
            dfSurvey=dfSurveyRaw.loc[:,['id','collection_type','date_type','origin_place_type','destin_place_type',
                                     'plan_for_trip','realtime_info','do_you_drive', 'used_veh_trip',
                                     'hh_member_travel','origin_transport','destin_transport','trip_in_oppo_dir',
                                     'oppo_dir_trip_time','gender_male', 'gender_female','race_white','resident_visitor',
                                     'work_location','student_status','english_ability', 'your_age','income', 'have_disability']]
            dfSurvey.columns=['id','season','dayofweek','purO','purD','plan','realtime','candrive','cdhvusdveh',
                              'HHcomp','access','egress','oppo','oppotime','male','female','white','visitor','worktype','stu',
                              'eng','age','income','disability']
    keygen=dict()
    for col in dfSurvey.columns[1:]:
        elems=dfSurvey.loc[:,col].unique()
        if len(elems)<50:
            keygen[col]=elems.tolist()
    keygen=pd.Series(keygen, name='orilevel').rename_axis('field').explode().reset_index()
    keygen.to_clipboard(index=False,header=False)
    dfConversion=dfConversion.loc[(dfConversion.version==ver) & (dfConversion.step=='post'),:]
    for fld in dfConversion.loc[:,'field'].unique():
        dfSlice=dfConversion.loc[dfConversion.field==fld,['orilevel','newlevel']]
        dfSlice=pd.Series(dfSlice['newlevel'].values,index=dfSlice['orilevel'])
        dfSurvey=dfSurvey.replace({fld:dfSlice})
    dfSurvey.fillna(value={'plan':'web','HHcomp':'0'},inplace=True)
    #plan nan ->phone planned
    # HHcomp nan ->0
    #can N  or NA (init)-> dependent
    #can Y cdv else -> potential
    #can Y cdv Y -> choice
    #worktype nan ->unemp
    #nonbinary
    #eng nan -> fluent


    dfSurvey.hr=dfSurvey.hr.astype(float)
    dfSurvey.dtypes
    dfSurvey=dfSurvey.sort_values(['sdate','hr','id']).reset_index(drop=True)
    return dfSurvey



class LatentClassNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LatentClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

class CombinedModel(nn.Module):
    def __init__(self, segmentation_input_size, num_classes, numeric_attr_size):
        super(CombinedModel, self).__init__()
        self.latent_nn = LatentClassNN(segmentation_input_size, num_classes)
        self.beta = nn.Parameter(torch.randn(num_classes, numeric_attr_size + 1))  # Add 1 for intercepts

    def forward(self, segmentation_bases, numeric_attrs):
        latent_probs = self.latent_nn(segmentation_bases)
        numeric_attrs_with_intercept = torch.cat(
            [numeric_attrs, torch.ones(numeric_attrs.size(0), 1).to(device)], dim=1
        )  # Add intercept term
        weighted_numeric = torch.matmul(latent_probs, self.beta)
        logit = torch.sum(weighted_numeric * numeric_attrs_with_intercept, dim=1)
        return torch.sigmoid(logit)

# Example usage
# Assume segmentation_bases, numeric_attrs, and y are your dataset components
segmentation_bases = segmentation_bases.to(device)
numeric_attrs = numeric_attrs.to(device)
y = y.to(device)

segmentation_input_size = segmentation_bases.shape[1]
numeric_attr_size = numeric_attrs.shape[1]
num_classes = 3  # You can choose the number of latent classes

model = CombinedModel(segmentation_input_size, num_classes, numeric_attr_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
l2_lambda = 0.01  # L2 regularization strength

losses = []  # List to store loss values

# Training loop
for epoch in range(100):  # number of epochs
    model.train()
    optimizer.zero_grad()
    outputs = model(segmentation_bases, numeric_attrs)
    loss = criterion(outputs, y)
    
    # L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg

    loss.backward()
    optimizer.step()

    losses.append(loss.item())  # Store the loss value

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Plot the loss values
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Inspect the estimated beta values
beta_values = model.beta.detach().cpu().numpy()
print("Estimated beta values (including intercepts):")
print(beta_values)
