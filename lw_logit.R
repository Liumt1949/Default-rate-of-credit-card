# ��һ�ι���LRģ��   
glm1 <- glm(traindata$default.payment.next.month~ .,traindata, family=binomial())  
summary(glm1) 
# �ڶ��ι���LRģ�ͣ��޳�Pֵ����0.05�ı���  
lraic=step(glm1)

predlr<- predict(lraic,testdata,type='response') # ѡ��typeΪresponse�򷵻���Ӧ������Ԥ����ʣ�ֵ��0-1֮��  
pred <- data.frame(predict(logit.aic,newtraindata,type='response')) 
pred<-as.numeric(pred)
# �����������  
confusion <- data.frame(predlr)  
confusion$pred <- ifelse(predlr>0.5,1,0)  
preTablelr<- table(testdata$default.payment.next.month, confusion$pred)  
(accuracy<-sum(diag(preTablelr))/sum(preTablelr))



xgb_lr.train.modelroc <- roc(testdata$default.payment.next.month, predlr)  
plot(xgb_lr.train.modelroc,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),  
     grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE) 

# ѵ������ȷ�ȡ��ٻ���  
preTablelr[2,2]/(preTablelr[1,2]+preTablelr[2,2]) # ��ȷ��  
preTablelr[2,2]/(preTablelr[2,1]+preTablelr[2,2]) # �ٻ��� 