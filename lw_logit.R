# 第一次构造LR模型   
glm1 <- glm(traindata$default.payment.next.month~ .,traindata, family=binomial())  
summary(glm1) 
# 第二次构造LR模型，剔除P值大于0.05的变量  
lraic=step(glm1)

predlr<- predict(lraic,testdata,type='response') # 选定type为response则返回响应变量的预测概率，值在0-1之间  
pred <- data.frame(predict(logit.aic,newtraindata,type='response')) 
pred<-as.numeric(pred)
# 计算混淆矩阵  
confusion <- data.frame(predlr)  
confusion$pred <- ifelse(predlr>0.5,1,0)  
preTablelr<- table(testdata$default.payment.next.month, confusion$pred)  
(accuracy<-sum(diag(preTablelr))/sum(preTablelr))



xgb_lr.train.modelroc <- roc(testdata$default.payment.next.month, predlr)  
plot(xgb_lr.train.modelroc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),  
     grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE) 

# 训练集精确度、召回率  
preTablelr[2,2]/(preTablelr[1,2]+preTablelr[2,2]) # 精确度  
preTablelr[2,2]/(preTablelr[2,1]+preTablelr[2,2]) # 召回率 
