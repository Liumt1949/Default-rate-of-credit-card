if(!suppressWarnings(require("RSNNS"))){
  install.packages("RSNNS")
  require("RSNNS")#多层感知器神经网络
}
setwd("C:/Users/Zoey/Desktop")
data<-read.csv("UCI_Credit_Card.csv",header=T)

summary(data)
head(data1)
sum(is.na(data))
#特征工程

library(corrplot)
library(ggplot2)
corr <-cor(data1)
corrplot(corr=corr)

data1<-data[,c(2,7:12,25)]
#####################################神经网络##################################
install.packages("RSNNS")
library(Rcpp)
library(RSNNS)
#定义网络输入 

Values= data1[,1:7]

#定义网络输出，并将数据进行格式转换 

Targets = decodeClassLabels(data1[,8])

#从中划分出训练样本和检验样本 

data2 = splitForTrainingAndTest(Values, Targets, ratio=0.15)

#数据标准化 
data2 = normTrainingAndTestSet(data2)


#利用mlp命令执行前馈反向传播神经网络算法 

model = mlp(data2$inputsTrain, data2$targetsTrain, size=3,maxit=500, inputsTest=data2$inputsTest, targetsTest=data2$targetsTest) 
# 计算混淆矩阵
predmlp <- predict(model,data2$inputsTest,type = "class")
confusionMatrix(data2$targetsTrain,fitted.values(model))
preTablemlp<-confusionMatrix(data2$targetsTest,predmlp)

(accuracy<-sum(diag(preTablemlp))/sum(preTablemlp))

summary(model)
model
weightMatrix(model)
extractNetInfo(model)

par(mfrow=c(1,1))
plotIterativeError(model)

mlproc <- roc(data2$targetsTest,predmlp)  
plot(mlproc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),  
     grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE) 

# 训练集精确度、召回率  
preTablemlp[2,2]/(preTablemlp[1,2]+preTablemlp[2,2]) # 精确度  
preTablemlp[2,2]/(preTablemlp[2,1]+preTablemlp[2,2]) # 召回率 

