setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(dplyr) # but you have to load the packages for each R Session 
library(mlogit)

df<-read.csv("dfMNL.csv",header=TRUE)
df$id<-as.character(df$id)
df$mode=ifelse(df$tway==1,"Tway","Non")

dfm<-mlogit.data(df,shape="long",choice="match",alt.var="mode")

mymodel<-mlogit(match~aux+wt+iv+nTrans,dfm)
summary(mymodel)

beta_constraints <- matrix(0, nrow = 4, ncol = length(coef(mymodel)))
beta_constraints[1, match("aux", names(coef(mymodel)))] <- -1 
beta_constraints[2, match("wt", names(coef(mymodel)))] <- -1 
beta_constraints[3, match("iv", names(coef(mymodel)))] <- -1 
beta_constraints[4, match("nTrans", names(coef(mymodel)))] <- -1 
mymodel2 <- mlogit(match~aux+wt+iv+nTrans, data = dfm,betaConstraints = beta_constraints)
summary(mymodel2)





LL0<-log(0.5)*nrow(df)/2 #4: # of alternatives; 210: # of responds
1-mymodel$logLik[1]/LL0 #we attach [1] to eliminate the redundant annotation
fitted<-as.data.frame(mymodel$probabilities)
max.col(fitted)
fitted$assignment<-colnames(fitted)[max.col(fitted)]
fitted$surveyed<-df[df$choice=="yes","mode"]
sum(fitted$assignment==fitted$surveyed)/nrow(fitted)