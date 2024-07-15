# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:50 2024

@author: Kwangho Baek baek0040@umn.edu
"""
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
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

def InputProcessing(surFile,pathFile,ver,convFile,imputeCols=[]):
    '''for debugging
    surFile='survey'
    pathFile='paths'
    ver=2022
    convFile='dfConv'
    imputeCols=['duration']
    '''
    dfSurveyRaw=pd.read_csv(surFile+str(ver)+r'.csv',low_memory=False, encoding='ISO 8859-1')
    dfConversion=pd.read_csv(convFile+r'.csv')
    #ver=int(pd.to_datetime(dfSurveyRaw.survey_date).dt.year.median())
    match ver:
        case 2016:
            pass
        case 2022:
            dfSurvey=dfSurveyRaw.loc[:,['id','collection_type','date_type','origin_place_type','destin_place_type',
                                     'plan_for_trip','realtime_info','do_you_drive', 'used_veh_trip',
                                     'hh_member_travel','origin_transport','destin_transport','trip_in_oppo_dir',
                                     'oppo_dir_trip_time','gender_male', 'gender_female','race_white','resident_visitor',
                                     'work_location','student_status','english_ability', 'your_age','income', 'have_disability']]
            dfSurvey.columns=['id','season','dayofweek','purO','purD','plan','realtime','candrive','cdhvusdveh',
                              'HHcomp','access','egress','oppo','oppotime','male','female','white','visitor','worktype','stu',
                              'engflu','age','income','disability']
    #use keygen to generate dfConv.csv
    keygen=dict()
    for col in dfSurvey.columns[1:]:
        elems=dfSurvey.loc[:,col].unique()
        if len(elems)<50:
            keygen[col]=elems.tolist()
    keygen=pd.Series(keygen, name='orilevel').rename_axis('field').explode().reset_index()
    keygen.to_clipboard(index=False,header=False)
    #how to deal with missing values?
    dfSurvey.fillna(value={'plan':'web','HHcomp':'0','worktype':'unemp','engflu':'1'},inplace=True)
    #refactoring some categorical variables from the survey format to model-able
    dfConversion=dfConversion.loc[(dfConversion.version==ver) & (dfConversion.step=='post'),:]
    for fld in dfConversion.loc[:,'field'].unique():
        dfSlice=dfConversion.loc[dfConversion.field==fld,['orilevel','newlevel']]
        dfSlice=pd.Series(dfSlice['newlevel'].values,index=dfSlice['orilevel'])
        dfSurvey=dfSurvey.replace({fld:dfSlice})
        try:
            dfSurvey[fld]=dfSurvey[fld].astype(float)
        except ValueError:
            pass
    #some variables are need to be defined using multiple survey responses
    dfSurvey['choicerider']='dependent'
    dfSurvey.loc[dfSurvey.candrive=='Yes','choicerider']='potentially'
    dfSurvey.loc[dfSurvey.cdhvusdveh=='Yes','choicerider']='choicerider'
    dfSurvey['nonbinary']=1
    dfSurvey.loc[(dfSurvey['male']+dfSurvey['female'])>0,'nonbinary']=0
    dfSurvey['purpose']='HB'
    dfSurvey.loc[(dfSurvey['purO']!='Home') & (dfSurvey['purD']!='Home'),'purpose']='NHB'
    dfSurvey.loc[dfSurvey.purpose=='HB','purpose']+=(dfSurvey.loc[dfSurvey.purpose=='HB','purO']+dfSurvey.loc[dfSurvey.purpose=='HB','purD']).str.replace('Home','')#.str[0]
    dfSurvey.loc[dfSurvey.purpose=='HB','purpose']='HBO' #there is one instance whose O and D are both Home
    #move on to the path preprocessing; paths retrieved from the repository SchBasedSPwithTE_Pandas
    dfPathRaw=pd.read_csv(pathFile+str(ver)+r'.csv',low_memory=False, encoding='ISO 8859-1')
    dfPath=dfPathRaw.drop(columns=['detail','cost','line','nodes','snap','elapsed','TE','hr']).dropna(subset='routes')
    dfPath=dfPath.loc[dfPath.sid.isin(dfPath.loc[dfPath.match==1,'sid'].unique())]
    dfPath['tway']=0
    dfPath['ntiv']=dfPath['iv']-dfPath['tiv']
    dfPath.loc[dfPath.tiv>dfPath.ntiv,'tway']=1
    dfPath=dfPath.loc[dfPath.sid.isin(dfPath.loc[dfPath.tway==1,'sid'])]
    dfPath['elap']=dfPath.label_t-dfPath.realDep
    #pairing starts
    dfCT=dfPath.loc[(dfPath.tway==1) & (dfPath.match==1),:] #chosen transitway
    dfCT=dfCT.loc[dfCT.groupby(['sid'])['elap'].rank(method='first')==1].reset_index(drop=True)
    dfAN=dfPath.loc[(dfPath.sid.isin(dfCT.sid)) & (dfPath.tway==0) & (dfPath.match==0),:] #alternative nontransitway
    dfAN=dfAN.loc[dfAN.groupby(['sid'])['elap'].rank(method='first')==1].reset_index(drop=True)
    dfCT=dfCT.loc[dfCT.sid.isin(dfAN.sid),:]
    dfCN=dfPath.loc[(dfPath.tway==0) & (dfPath.match==1),:] #chosen nontransitway
    dfCN=dfCN.loc[dfCN.groupby(['sid'])['elap'].rank(method='first')==1].reset_index(drop=True)
    dfAT=dfPath.loc[(dfPath.sid.isin(dfCN.sid)) & (dfPath.tway==1) & (dfPath.match==0),:] #alternative Transitway
    dfAT=dfAT.loc[dfAT.groupby(['sid'])['elap'].rank(method='first')==1].reset_index(drop=True)
    dfCN=dfCN.loc[dfCN.sid.isin(dfAT.sid),:]
    dfPath=pd.concat([dfCT,dfAN,dfCN,dfAT]).sort_values(['sid','tway']).reset_index(drop=True)
    dfPath=dfPath.loc[dfPath.sid.isin(np.union1d(np.intersect1d(dfCT.sid,dfAN.sid),np.intersect1d(dfCN.sid,dfAT.sid))),:]
    if len(dfPath.sid.unique())*2 != len(dfPath):
        raise Exception('Pairing Failed')
    #Imputing activity duration
    dfSurvey=pd.merge(dfSurvey,dfPathRaw.loc[dfPathRaw.match==1,['sid','realDep']],left_on='id',right_on='sid')
    dfSurvey['duration']=dfSurvey.oppotime-dfSurvey.realDep/60
    dfSurvey=dfSurvey.drop_duplicates('id').drop(columns=['purO','purD','candrive','cdhvusdveh','oppotime','sid','realDep']).reset_index(drop=True)
    print(dfSurvey.isnull().sum())
    catcols=['season','dayofweek','plan','access','egress','worktype','stu','choicerider','purpose']
    enc=OneHotEncoder(sparse_output=False)
    dfOnehot=pd.DataFrame( enc.fit_transform(dfSurvey[catcols]),columns=enc.get_feature_names_out())
    dfSurvey=pd.concat([dfSurvey.drop(columns=catcols),dfOnehot],axis=1,ignore_index=False)
    if len(imputeCols)>0: #to be updated
        pass #KNNImputer(n_neighbors=20,weights='distance').fit_transform(dfSurvey.drop(columns='id'))
    else:
        pass # only keep complete info?
    #hardcoding now...
    dfSurvey['duration']=KNNImputer(n_neighbors=20,weights='distance').fit_transform(dfSurvey.drop(columns='id'))[:,12]
    #final path filtering
    pathfilter=dfPath.groupby('sid').agg({'tway':'sum','match':'sum','elap':['count','min','max']}).reset_index()
    pathfilter.columns=['sid','tway','match','count','mint','maxt']
    pathfilter['compDiff']=pathfilter.maxt-pathfilter.mint
    pathfilter['compProp']=pathfilter.maxt/pathfilter.mint
    pathfilter2=pathfilter.loc[(pathfilter.compDiff<10) | ((pathfilter.compProp<1.5)) ]
    len(pathfilter2)/len(pathfilter)
    dfSurvey=dfSurvey.loc[dfSurvey.id.isin(pathfilter2.sid.unique()),:]
    dfPath=dfPath.loc[dfPath.sid.isin(pathfilter2.sid.unique()),:]
    #dfSurvey[dfSurvey.drop(columns='duration').duplicated()]
    dfPath=dfPath.drop(columns=['ind','label_t','label_c','realDep','routes','elap']).rename(columns={"sid": "id"})
    return dfSurvey, dfPath

dfSurvey, dfPath= InputProcessing('survey','paths',2022,'dfConv',imputeCols=['duration'])



def genChoicedf(dfSurvey,dfPath):
    pass


def genNeuralInputs(dfSurvey,dfPath):
    dfMain=dfPath.loc[dfPath.tway==1,['id','match','nwk','wk','wt','ntiv','tiv','nTrans']]
    dfSub=dfPath.loc[dfPath.tway==0,['id','match','nwk','wk','wt','ntiv','tiv','nTrans']]
    if not np.all(dfMain.id.values==dfSub.id.values):
        raise Exception('organize dfPath or recheck the pairing steps')
    dfMain.iloc[:,2:]=dfMain.iloc[:,2:].to_numpy()-dfSub.iloc[:,2:].to_numpy()
    dfMain=pd.merge(dfSurvey,dfMain,on='id')
    chloc=np.where(dfMain.columns=='match')[0][0]
    seg=torch.from_numpy(dfMain.iloc[:,1:chloc].to_numpy(dtype='float32'))
    nume=torch.from_numpy(dfMain.iloc[:,(chloc+1):].to_numpy(dtype='float32'))
    ch=torch.from_numpy(dfMain.iloc[:,chloc].to_numpy(dtype='float32'))
    return seg, nume, ch
    
class LatentClassNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LatentClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

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

segmentation_bases, numeric_attrs, y=genNeuralInputs(dfSurvey,dfPath)
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
for epoch in range(300):  # number of epochs
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
