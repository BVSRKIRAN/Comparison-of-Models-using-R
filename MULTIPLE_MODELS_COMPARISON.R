library(caret)
library(Metrics)
library(pROC)
library(ROCR)

Data <- read.csv(".../GermanCredit.csv", sep = ",", header = T)
head(Data)

table(Data$Class)

Data$Class1[Data$Class == "Good"] <- 1
Data$Class1[Data$Class == "Bad"] <- 0

names(Data)
Data <- Data[,-10]

Data$Class1 <- as.factor(Data$Class1)

set.seed(999)
ind <- sample(2, nrow(train), replace=T, prob=c(0.70,0.30))
trainData<-Data[ind==1,]
testData <- Data[ind==2,]

### GLM 

set.seed(33)
Logfit <- train(Class1 ~ .,data = trainData[,-10],method = "glm", metric="Accuracy")
names(Logfit)
confusionMatrix(Logfit)
PRED_LOG <- predict(Logfit,testData)

### Decision Trees 

set.seed(33)
DTREE <- train(Class1 ~ .,data = trainData[,-10],method = "ctree", metric="Accuracy")
confusionMatrix(DTREE)
PRED_DTREE <- predict(DTREE,testData)

### Random Forest 

set.seed(33)
RFOREST <- train(Class1 ~ .,data = trainData[,-10],method = "rf", metric="Accuracy")
confusionMatrix(RFOREST)
PRED_RFOREST <- predict(RFOREST,testData)

### KNN 

set.seed(33)
KNN_MOD <- train(Class1 ~ .,data = trainData[,-10],method = "knn", metric="Accuracy")
confusionMatrix(KNN_MOD)
PRED_KNN <- predict(KNN_MOD,testData)

### Naive Bayes

set.seed(33)
NB_MOD <- train(Class1 ~ .,data = trainData[,-10],method = "naive_bayes", metric="Accuracy")
confusionMatrix(NB_MOD)
PRED_NB <- predict(NB_MOD,testData)

### SVM 

set.seed(33)
SVM_MOD <- train(Class1 ~ .,data = trainData[,-10],method = "svmLinear3", metric="Accuracy")
confusionMatrix(SVM_MOD)
PRED_SVM <- predict(SVM_MOD,testData)

### GBM 

set.seed(33)
GBM_MOD <- train(Class1 ~ .,data = trainData[,-10],method = "gbm", metric="Accuracy", verbose=F)
confusionMatrix(GBM_MOD)
PRED_GBM <- predict(GBM_MOD,testData)

### XGBM 

set.seed(33)
XGBM_MOD <- train(Class1 ~ .,data = trainData[,-10],method = "xgbTree", metric="Accuracy", verbose=F)
confusionMatrix(XGBM_MOD)
PRED_XGBM <- predict(XGBM_MOD,testData)


# List of predictions
preds_list <- list(as.numeric(PRED_LOG), as.numeric(PRED_DTREE),
                   as.numeric(PRED_RFOREST),as.numeric(PRED_KNN),
                   as.numeric(PRED_NB),as.numeric(PRED_SVM),
                   as.numeric(PRED_GBM), as.numeric(PRED_XGBM))

# List of actual values (same for all)
m <- length(preds_list)
actuals_list <- rep(list(as.numeric(testData$Class1)), m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves",  pch=19,
     cex.main=1.25, cex.lab=1.5, cex.axis=0.75)
legend(x = "bottomright", 
       legend = c("Logistic Reg","D Trees", "Random Forest", "KNN",
                  "Naive Bayes", "SVM", "GBM", "XGBM"),
       fill = 1:m, ncol = 2,
       cex = 0.6)

