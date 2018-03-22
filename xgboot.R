if(!suppressWarnings(require("xgboost"))){
  install.packages("xgboost")
  require("xgboost")
}

library(xgboost)  
ind<-sample(2,nrow(data1),replace=TRUE,prob=c(0.7,0.3)) #对数据分成两部分，70%训练数据，30%检测数据  
traindata<- data1 [ind==1,]  #训练集  
testdata<- data1[ind==2,]  #测试集  
traindatax=as.matrix(traindata[,c(1:7)])  
traindatay=as.matrix(traindata[,8])  
testdatax=as.matrix(testdata[,c(1:7)])  
testdatay=as.matrix(testdata[,8]) 

bst <- xgboost(data = traindatax, label = traindatay, max.depth = 2, eta = 1,nround = 2) 
pred <- predict(bst, testdatax)  
#table(pred, testdatay)  
table(round(pred,0), testdatay) 
xgb.save(bst, 'model.save')  
bst = xgb.load('model.save')  
  
#做交叉验证的函数参数与训练函数基本一致，只需要在原有参数的基础上设置nfold：  
cv.res <- xgb.cv(data = traindatax, label = traindatay, max.depth = 2, eta = 1, nround = 2, objective = "binary:logistic", nfold = 5) 

  
dtrain <- xgb.DMatrix(data =traindatax, label = traindatay) # 构造模型需要的xgb.DMatrix对象，处理对象为稀疏矩阵  

param <- list(max_depth=2, eta=1, silent=1, objective='binary:logistic') # 定义模型参数  
nround = 4  

bst = xgb.train(params = param, data =dtrain, nrounds = nround, nthread = 2) # 构造xgboost模型  
accuracy.after <- sum((predict(bst, testdatax) >= 0.5) == testdatay) /
  length(testdatay)
# 利用xgboost包的xgb.create.features构造新特征变量
new.features.train <- xgb.create.features(model = bst, traindatax) # 生成xgboost构造的新特征组合，训练集  
new.features.test <- xgb.create.features(model = bst, testdatax) # 生成xgboost构造的新特征组合，测试集  



# learning with new features
new.dtrain <- xgb.DMatrix(data = new.features.train, label = traindatay)
new.dtest <- xgb.DMatrix(data = new.features.test, label = testdatay)
watchlist <- list(train = new.dtrain)
bst <- xgb.train(params = param, data = new.dtrain, nrounds = nround, nthread = 2)

# Model accuracy with new features
accuracy.after <- sum((predict(bst, new.dtest) >= 0.5) == testdatay) /
  length(testdatay)

newdtrain <- as.data.frame(as.matrix(new.features.train)) # 将训练集的特征组合转化为dataframe格式  
newdtest <- as.data.frame(as.matrix(new.features.test)) # 将测试集的特征组合转化为dataframe格式  

newtraindata <- cbind(newdtrain,backflag1=traindatay) # 将训练集的自变量和因变量合并  
newtestdata <- cbind(newdtest,backflag1=testdatay) # 将测试集的自变量和因变量合并  

model <- xgb.dump(bst,with_stats = T) # 显示计算过程，查看树结构  
model   
names <- dimnames(data.matrix(traindata[,c(1:7)]))[[2]] # 获取特征的真实名称  
importance_matrix <- xgb.importance(names,model=bst) # 计算变量重要性  
xgb.plot.importance(importance_matrix[,])  



head(newtraindata)

# 第一次构造LR模型   
fit <- glm(backflag1~ .,newtraindata, family=binomial())  
summary(fit)  

# 第二次构造LR模型，剔除P值大于0.05的变量  
logit.aic=step(fit)
summary(logit.aic)
# 对训练集进行预测  
pred <- predict(logit.aic,newtraindata,type='response') # 选定type为response则返回响应变量的预测概率，值在0-1之间  
pred <- data.frame(predict(logit.aic,newtraindata,type='response')) 
pred<-as.numeric(pred)
library(pROC)  
xgb_lr.train.modelroc <- roc(newtraindata$backflag1, pred)  
plot(xgb_lr.train.modelroc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),  
     grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE) 

auc <- performance(predict, measure = "auc")  
auc <- auc@y.values[[1]]  
auc  

# 计算混淆矩阵  
confusion <- data.frame(pred)  
confusion$pred <- ifelse(confusion$pred>0.5,1,0)  
xgb_lr.train.result <- table(newtraindata$backflag1, confusion$pred)  
xgb_lr.train.result  

# 计算准确率  
(xgb_lr.train.result[1,1]+xgb_lr.train.result[2,2])/nrow(newtraindata)  

# 训练集精确度、召回率  
xgb_lr.train.result[2,2]/(xgb_lr.train.result[1,2]+xgb_lr.train.result[2,2]) # 精确度  
xgb_lr.train.result[2,2]/(xgb_lr.train.result[2,1]+xgb_lr.train.result[2,2]) # 召回率  





