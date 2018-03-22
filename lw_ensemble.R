# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)
# Load the dataset
setwd("C:/Users/Zoey/Desktop")
data<-read.csv("UCI_Credit_Card.csv",header=T)
data1<-data[,c(2,7:12,25)]
head(data1)
data1$default.payment.next.month <- as.factor(data1$default.payment.next.month)

library(randomForest)
# Example of Bagging algorithms
## method重抽样方法
## number交叉验证折数
## repeatsor重抽样迭代次数
control <- trainControl(method="repeatedcv", number=5, repeats=3)
metric <- "Accuracy"
# Bagged CART
set.seed(111)
fit.treebag <- train(default.payment.next.month~., data=data1, method="treebag", metric=metric, trControl=control)
fit2<- train(default.payment.next.month~., data=data1, method="gbm", metric=metric, trControl=control)
fit.NET <- train(default.payment.next.month~., data=data1, method="pcaNNett", metric=metric, trControl=control)
fit1 <- caret::train(default.payment.next.month~., data=data1, method='rf', metric=metric, trControl=control)
# summarize results
boosting_results <- resamples(list(gbm=fit2, net=fit1))
summary(boosting_results)
dotplot(boosting_results)

# Example of Stacking algorithms
# create submodels
levels(data1$default.payment.next.month) <- make.names(levels(factor(data1$default.payment.next.month)))
control <- trainControl(method="repeatedcv", number=5,repeats=3, savePredictions=T, classProbs=TRUE)
algorithmList <- c('gbm', 'glm','nnet','')
set.seed(111)
models <- caretList(default.payment.next.month~., data=data1,trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)
models$rpart<-NULL

# stack using glm
set.seed(seed)
stackControl <- trainControl(method="repeatedcv", number=5, repeats=3, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)


