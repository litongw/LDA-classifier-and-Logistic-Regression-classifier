# load package
library(MASS)
# load the data
train.3<-read.table("train.3.txt", header=FALSE, sep=",") # training set for 3s
train.5<-read.table("train.5.txt", header=FALSE, sep=",") # training set for 5s
train.8<-read.table("train.8.txt", header=FALSE, sep=",") # training set for 8s
xtrain<-rbind(as.matrix(train.3),as.matrix(train.5),as.matrix(train.8))
ytrain<-rep(c(3,5,8),c(nrow(train.3),nrow(train.5),nrow(train.8)))
test<-as.matrix(read.table("zip.test")) # test set 
ytest<-test[,1]
xtest<-test[ytest==3|ytest==5|ytest==8,-1]
ytest<-ytest[ytest==3|ytest==5|ytest==8]
# store training and test errors
error<-matrix(0,nrow=4,ncol=2)
# 1. LDA on the original 256 dimensional space
lda.orig<-lda(xtrain,ytrain)
error[1,1]<-sum(predict(lda.orig)$class!=ytrain)/length(ytrain)
error[1,2]<-sum(predict(lda.orig,xtest)$class!=ytest)/length(ytest)
# 2. LDA on the leading 49 principle components of the features
# center and scale
xtrain.c<-apply(xtrain,2,mean)
xtrain.s<-scale(xtrain,center=xtrain.c,scale=F)
xtest.s<-scale(xtest,center=xtrain.c,scale=F)
# find leading 49 pc of the feature
v<-svd(xtrain.s)$v[,1:49]
train.pc<-xtrain.s %*% v
test.pc<-xtest.s %*% v
lda.pc<-lda(train.pc,ytrain)
error[2,1]<-sum(predict(lda.pc)$class!=ytrain)/length(ytrain)
error[2,2]<-sum(predict(lda.pc,test.pc)$class!=ytest)/length(ytest)
# 3. LDA on filtered data
# build the function average each non-overlapping 2*2 pixel block
filter<-function(x){
  x<-matrix(x,16,16)
  seq<-rep(1:2,8)
  x<-x[seq==1,]+x[seq==2,]
  x<-x[,seq==1]+x[,seq==2]
  as.vector(x)/4
}
train.f<-t(apply(xtrain,1,filter))
test.f<-t(apply(xtest,1,filter))
lda.filter<-lda(train.f,ytrain)
error[3,1]<-sum(predict(lda.filter)$class!=ytrain)/length(ytrain)
error[3,2]<-sum(predict(lda.filter,test.f)$class!=ytest)/length(ytest)
# 4. Multiple linear logistic regression
library(glmnet)
log.reg<-glmnet(train.f,factor(ytrain),family="multinomial")
error[4,1]<-sum(as.numeric(predict(log.reg,train.f,s=log.reg$lambda[93],type="class"))!=ytrain)/length(ytrain)
error[4,2]<-sum(as.numeric(predict(log.reg,test.f,s=log.reg$lambda[93],type="class"))!=ytest)/length(ytest)
# compare the procedures with respect to training and test misclassification error
error<-as.data.frame(round(error,4),row.names = c("LDA on original 256 dimensional data","LDA on the leading 49 principal components","LDA on filtered data","Multiple linear logistic regression on filtered data"))
colnames(error)<-c("Training Error","Test Error")
print(error)
