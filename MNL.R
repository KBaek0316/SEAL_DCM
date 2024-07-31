setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(dplyr) # but you have to load the packages for each R Session 
library(mlogit)
library(ggplot2)
library(ggpattern)

df<-read.csv("./Results/TRB1/dfMNL_ivadj.csv",header=TRUE)
df$id<-as.character(df$id)
df$mode=ifelse(df$tway==1,"Tway","Non")

dfNNLCCM<-read.csv("./Results/TRB1/modelout_ivadj_auxwtivntranst.csv",header=TRUE)
dfNNLCCM[is.na(as.numeric(dfNNLCCM[,1])),c('tr')]<-NA
trid<-dfNNLCCM[dfNNLCCM$tr==1,1] #first col: X = header
trid<-trid[!is.na(trid)]
tsid<-dfNNLCCM[dfNNLCCM$tr==0,1]
tsid<-tsid[!is.na(tsid)]


dfm<-mlogit.data(df,shape="long",choice="match",alt.var="mode")

mymodel<-mlogit(match~aux+wt+iv+nTrans,dfm[dfm$id %in% trid,])
summary(mymodel)
tspred<-predict(mymodel,newdata=dfm[dfm$id %in% tsid,])
tsy<-matrix(df[df$id %in% tsid,c('match')],ncol=2,byrow=TRUE,dimnames=list(c(tsid),c('Non','Tway'))) 
print(1-(sum(tsy*log(tspred))/sum(tsy*log(c(mean(tsy[,1]),mean(tsy[,2])))))) #rhosq


mymodel2<-mlogit(match~ov+iv+nTrans,dfm[dfm$id %in% trid,])
summary(mymodel2)
tspred<-predict(mymodel2,newdata=dfm[dfm$id %in% tsid,])
tsy<-matrix(df[df$id %in% tsid,c('match')],ncol=2,byrow=TRUE,dimnames=list(c(tsid),c('Non','Tway'))) 
print(1-(sum(tsy*log(tspred))/sum(tsy*log(c(mean(tsy[,1]),mean(tsy[,2])))))) #rhosq




dfNNLCCM2<-read.csv("./Results/TRB1/modelout_ivadj_ovivntrans.csv",header=TRUE)
dfNNLCCM2[is.na(as.numeric(dfNNLCCM2[,1])),c('tr')]<-NA


infomat<-dfNNLCCM[is.na(dfNNLCCM$tr),2:(ncol(dfNNLCCM)-1)]
row.names(infomat)<-dfNNLCCM[is.na(dfNNLCCM$tr),1]
apply(infomat,1,mean)
apply(infomat,1,sd)
infomat[,which(infomat[2,]==max(infomat[2,]))] #max rho_ts



infomat2<-dfNNLCCM2[is.na(dfNNLCCM2$tr),2:(ncol(dfNNLCCM2)-1)]
row.names(infomat2)<-dfNNLCCM2[is.na(dfNNLCCM2$tr),1]
#infomat2<-infomat2[,which(infomat2[2,]>=quantile(t(infomat2[2,]), 2/3))]
memmat2<-dfNNLCCM2[!is.na(dfNNLCCM2$tr),2:(ncol(dfNNLCCM2)-1)]
apply(infomat2,1,mean)
apply(infomat2,1,sd)

infomat2[,which(infomat2[2,]==max(infomat2[2,]))] #best model
mean(round(memmat2[,which(infomat2[2,]==max(infomat2[2,]))])) #Class1 membership prop of best model
#round(memmat2[,which(infomat2[2,]==max(infomat2[2,]))]) #class assignments of best model

df2<-df %>% filter(match==1) %>% select(-realDep,-routes,-mode,-id,-match)
df2$aclass<-round(apply(memmat2,1,mean))

df2$iv<--1*df2$iv
df2[df2$aclass==0,c('aclass')]<-2
df2$aclass<-as.factor(df2$aclass)
summary(df2$aclass)
df2<-df2 %>% select(aclass,everything())

#dfsummary<-df2 %>% group_by(aclass) %>% summarize_all(list(mean=mean,sd=sd))
dfmean<-df2 %>% group_by(aclass) %>% summarize_all(mean)
dfmean$aclass<-paste0('mean_',dfmean$aclass)
dfsd<-df2 %>% group_by(aclass) %>% summarize_all(sd)
dfsd$aclass<-paste0('sd_',dfsd$aclass)
dfstore<-dfmean
dfstore[]<-NA
dfstore[,1]<-c("tstat",'Pval')

for (i in 2:ncol(dfstore)){ #from nTrans to aclass
    tres<-t.test(df2[df2$aclass==1,i],df2[df2$aclass==2,i])
    dfstore[1,i]<-tres$statistic
    dfstore[2,i]<-tres$p.value
    for (j in 1:ncol(infomat2)){
        dftemp<-as.data.frame(cbind(df2[,i],2-round(memmat2[,j])))
        colnames(dftemp)<-c('meanval','aclass')
        dfdistfrag<-dftemp %>% group_by(aclass) %>% summarize_all(mean)
        dfdistfrag$frommodel<-j
        dfdistfrag$var<-colnames(df2)[i]
        if (i*j==2){
            dfdist<-dfdistfrag
        }
        else{
            dfdist<-rbind(dfdist,dfdistfrag)
        }
    }
}
dffinal<-rbind(dfmean,dfsd,dfstore)
dffinal<-rbind(dffinal,dffinal[6,]<0.05)
dffinal[7,1]<-"sig"

write.csv(dffinal,paste0("./Results/TRB1/stats_",sum(df2$aclass==1),"_",sum(df2$aclass==2),".csv"),row.names=FALSE)
write.csv(dfdist,"./Results/TRB1/bymodels.csv",row.names=FALSE)


dffin<-read.csv("./Results/TRB1/bymodels.csv",header=TRUE)
plotvars<-c("dayofweek_Weekday","realtime","visitor","purpose_HBW","purpose_HBEd","choicerider_choicerider","worktype_fullO","worktype_fullT","worktype_partialT")
dfnames<-data.frame(var=plotvars,variables=c("Weekday","Real Time\nInfo User","Twin Cities\nVisitor","Home Based\nWork Trip","Home Based\nSchool Trip","Transit\nChoice Rider","Full\nOffice Worker","Full\nTeleworker","Hybrid\nTeleworker"))
dfnames$variables<-factor(dfnames$variables,levels=c("Weekday","Real Time\nInfo User","Twin Cities\nVisitor","Home Based\nWork Trip","Home Based\nSchool Trip","Transit\nChoice Rider","Full\nOffice Worker","Full\nTeleworker","Hybrid\nTeleworker"))

dfplot<-dffin[dffin$var %in% plotvars,]
dfplot<-left_join(dfplot,dfnames,by="var")
dfplot$AssignedClass<-"Class 1: More likely\nto use Transitway"
dfplot[dfplot$aclass==2,"AssignedClass"]<-"Class 2: Less likely\nto use Transitway"


#dfplot$AssignedClass<-factor(paste0("Class ",dfplot$aclass))

ggplot(dfplot,aes(y=variables,x=meanval))+ylab("Variables")+
    xlab("Each Class's Composition of People of the Variable Being Inspected")+
    geom_boxplot_pattern(aes(pattern=AssignedClass,fill=AssignedClass),pattern_spacing=0.01,pattern_density=0.4)+
    scale_pattern_manual(values= c("none","pch"))+
    scale_fill_manual(values= c("#4E95D9","#F2AA84"))+
    theme(legend.position = c(0.99, .99), legend.justification = c(1, 1),
          legend.key = element_blank(),
          legend.key.width= unit(.5, 'cm'),legend.key.height= unit(.5, 'cm'),)
ggsave("./Results/TRB1/dist.png",width=6.5,height=4.5,units="in",dpi=300)    


