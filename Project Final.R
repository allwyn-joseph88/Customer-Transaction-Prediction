########################################################################################################
########################################################################################################
########################################################################################################

rm(list=ls(all=T))
setwd("F:/Edwisor Project/Santander Project")
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart",'mlr', "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','dplyr','tidyverse','ggthemes','data.table',
      'speedglm','outliers','psych','scales','VIM','ROCR','pROC','xgboost','rpart')

#install.packages()
lapply(x, require, character.only = TRUE)
rm(x)
########################################################################################################
########################################################################################################
########################################################################################################

train_data = read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))
train=train_data
test_data= read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))
test=test_data

###########################################################################################################
###########################################Explore the data############################################
###########################################################################################################

glimpse(train[1:10, 1:10])
glimpse(test[1:10, 1:10])
dim_desc(train)
dim_desc(test)

##CHECKING  FOR DUPLICATE COLLUMNS
dup<-function(x){if(length(unique(colnames(x))==ncol(x))){print('No')}else{print('Yes')}}
cat('Is there any duplicate column in train data:', dup(train), 
    '\nIs there any duplicate column in test data:', dup(test), sep = ' ')  

##no of datatypes
cat('Number of factor columns in train dataset:',length(which(sapply(train, is.factor))),'\nNumber of numeric columns in train dataset:',length(which(sapply(train, is.numeric))))

#checking for duplicate rows in ID_code
train$ID_code %>% unique %>% length
test$ID_code %>% unique %>% length

summary(train)
summary(test)

str(train,list.len=ncol(train))
str(test,list.len=ncol(test))
#######################################################################################################
#########################################Missing Value Analysis########################################
#######################################################################################################
sum(is.na(train))
colSums(is.na(train))
sum(is.na(test))
colSums(is.na(test))
#There are no missing values in both train and test data.
########################################################################################################
#########################################Target Variable################################################
#######################################################################################################
#Distribution of target for customers?Histogram
#ggplot(data = train, aes(target))+
#  geom_histogram(bins=4,color="darkblue",fill="lightblue")+ xlab('Target')+
#  ggtitle('Distribution of Target Variable')

## alternate histogram
target_df <- data.frame(table(train$target))
colnames(target_df) <- c("target", "freq")
ggplot(data=target_df, aes(x=target, y=freq, fill=target)) +
  geom_bar(position = 'dodge', stat='identity', alpha=0.5) +
  scale_fill_manual("legend", values = c("1" = "dodgerblue", "0"="firebrick1")) +
  theme_classic()


# data is imbalanced

#train_data_0 = train %>% filter(target == 0)
#train_data_1 = train %>% filter(target == 1)


#10.04% is 0  ## #89.95% is 0.
# implies that the dataset is very unbalanced.

#  distribution of target box plot
#ggplot(data = train, aes(x = "", y = target)) + 
  #geom_boxplot()+ylab('Target') +ggtitle('Distribution of Target-Boxplot')

##From box plot we confirm the 

#ggplot(data = train, aes(log(target)))+
#  geom_histogram(bins=150,color="black", fill="pink")+ xlab('Target')+
#  ggtitle('Normal Distribution of Target-Histogram')

summary(train$target)
#######################################################################################################
#########################################Pre Processing################################################
########################################################################################################
########################################################################################################

############################################Outlier Analysis#############################################

train$target=as.factor((train$target))
############################################Train Data Outlier Analysis################################################3

cnames <- names(train)[c(3:202)]
plot_list = list()

for (i in cnames){
  plot1<-ggplot(data=train, aes_string( x=train$target, y=train[[i]], 
                                        fill=train$target)) +
    stat_boxplot(geom = "errorbar", width = 0.5)+
    geom_boxplot(outlier.colour = "red", outlier.size = 3)+
    scale_y_continuous(breaks = scales::pretty_breaks(15)) + 
    guides(fill=FALSE) + theme_bw() + ggtitle(paste("Outlier Analysis of",i,'in Train')) +
    xlab("Target") + ylab(i)
  plot_list[[i]] = plot1
}

#  create pdf where each page is a separate plot.
pdf("Boxplots of Train.pdf")
for (i in cnames) {
  print(plot_list[[i]])
}
dev.off()
############################################Train Data Outlier Removal##################################
cnames <- names(train)[c(3:202)]
plot_list = list()

#Replace all outliers with NA and impute

for(i in cnames){
  val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  train[,i][train[,i] %in% val] = NA
}
# 
############################################Train Data Outlier Imputation####################################

sum(is.na(train)) #26533 
colSums(is.na(train))
dim(train)
for (i in cnames) {
  train[,i][is.na(train[,i])] = mean(train[,i], na.rm = T)
}
########################################################################################################
########################################################################################################
########################################################################################################
str(train,list.len=ncol(train))
numeric_index = sapply(train,is.numeric)
numeric_train = train[,numeric_index]


#########################################Feature Selection#############################################
####################################Correlations######################################################
###########To find the correlation between a set of variables, use the cor command:####################


a=round(cor(numeric_train),2)    #round the correlations to 2 decimals
results = print(a)

summary(a[upper.tri(a)])

##Correlations between features nearly zero.
#hence well have to consider all the features.

######################################################################################################
###############################################FEATURE SCALING########################################
###################################normality check for train#############################################
cnames <- names(train)[c(3:202)]
plot_list = list()

for (i in cnames){
  plot3<-ggplot(train, aes(x=train[[i]], fill=as.factor(train$target))) +
    geom_histogram(bins=500, alpha=.5, position="identity") + ggtitle(i)
  plot_list[[i]] = plot3
}

# Another option: create pdf where each page is a separate plot.
pdf("Histograms before Normalisation-train.pdf")
for (i in cnames) {
  print(plot_list[[i]])
}
dev.off()


range(train[3:202])

##################################normalisation fr train################################################
cnames <- names(train)[c(3:202)]
plot_list = list()

for(i in cnames){
  print(i)
  train[,i] = (train[,i] - min(train[,i]))/
    (max(train[,i] - min(train[,i])))
}

range(train[34])
trainT=train
#write.csv(train,"train1.csv", row.names = FALSE)
#train$target<-as.numeric(as.factor(train$target))
##########################################Sampling######################################################

#Divide data into train and test using stratified sampling method

set.seed(123)
train.index = createDataPartition(train$target, p = .70, list = FALSE)
train1 = train[ train.index,]
validate  = train[-train.index,]
summary(train1$target)
summary(validate$target)
train$ID_code<-NULL
validate$ID_code<-NULL

########################################################################################################
rm(plot_list)
rm(test_data)
rm(train_data)
rm(cnames)
rm(val)

#Logistic Regression
logit_model = glm(target ~ ., data = train1, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = validate, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
logit_Predictions1=as.factor(logit_Predictions)

##Evaluate the performance of classification model
ConfMatrix_LR = table(validate$target, logit_Predictions1)

confusionMatrix(logit_Predictions1, validate$target)

logit_Predictions2=as.numeric(logit_Predictions)
pred <- prediction(logit_Predictions2, validate$target)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
auc_ROC <- performance(pred, measure = "auc")
auc_ROC <- auc_ROC@y.values[[1]]



########################################################################################################
########################################################################################################
########################################################################################################

###Random Forest

RF_model = randomForest(target ~ ., train1, importance = TRUE, ntree = 100)


#Extract rules fromn random forest
#transform rf object to an .inTrees' format
# treeList = RF2List(RF_model)  
# 
# #Extract rules
# exec = extractRules(treeList, train[,-201])  # R-executable conditions
# 
# #Visualize some rules
# exec[1:2,]
# 
# #Make rules more readable:
# readableRules = presentRules(exec, colnames(train))
# readableRules[1:2,]
# 
# #Get rule metrics
# ruleMetric = getRuleMetric(exec, train[,-201], train$target)  # get rule metrics
# 
# #evaulate few rules
# ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, validate[,-1])

##Evaluate the performance of classification model
ConfMatrix = table(validate$target, RF_Predictions)
confusionMatrix(ConfMatrix)


RF_Predictions=as.numeric(RF_Predictions)
pred1 <- prediction(RF_Predictions, validate$target)
perf <- performance(pred1,"tpr","fpr")
plot(perf,colorise=TRUE)
auc_ROC <- performance(pred, measure = "auc")
auc_ROC <- auc_ROC@y.values[[1]]

########################################################################################################
########################################################################################################
########################################################################################################
# Read the input data
trainX <- fread("train.csv")
testX <- fread("test.csv")

trainY <- trainX$target
trainX <- trainX[, !c("target", "ID_code"), with = F]
testX <- testX[, !c("ID_code"), with = F]

# Holdout 20% of train data for model validation
inTrain <- createDataPartition(factor(trainY), p = 0.7, list = F)
trainX_train <- trainX[inTrain]
trainX_valid <- trainX[-inTrain]
trainY_train <- trainY[inTrain]
trainY_valid <- trainY[-inTrain]

# preparing XGB matrix
dtrain <- xgb.DMatrix(data = as.matrix(trainX_train), label = as.matrix(trainY_train))
dvalid <- xgb.DMatrix(data = as.matrix(trainX_valid), label = as.matrix(trainY_valid))

# parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.1,
  gamma = 80,
  max_depth = 7,
  min_child_weight = 1, 
  subsample = 0.5,
  colsample_bytree = 0.5,
  scale_pos_weight = round(sum(!trainY) / sum(trainY), 2))


set.seed(123)
xgb_model <- xgb.train(
  params = params, 
  data = dtrain, 
  nrounds = 1000, 
  watchlist = list(train=dtrain, val=dvalid),
  print_every_n = 10, 
  early_stopping_rounds = 50, 
  maximize = T, 
  eval_metric = "auc")

#model prediction
xgbpred <- predict(xgb_model, dvalid)
xgbpred <- ifelse(xgbpred > 0.5,1,0)
confusionMatrix(factor(xgbpred), factor(trainY_valid))

#view variable importance plot
imp_mat <- xgb.importance(feature_names = colnames(trainX_train), model = xgb_model)
xgb.plot.importance(importance_matrix = imp_mat[1:10])

pred_sub <- predict(xgb_model, newdata=as.matrix(testX), type="response")

pred_sub=as.numeric(pred_sub)
pred1 <- prediction(pred_sub, validate$target)
perf <- performance(pred1,"tpr","fpr")
plot(perf,colorise=TRUE)
auc_ROC <- performance(pred, measure = "auc")
auc_ROC <- auc_ROC@y.values[[1]]

finalpredictions <- as.data.frame(pred_sub)
names(finalpredictions)=c('target')
write.csv(finalpredictions, file="Final_Prediction__XgBoost.csv", row.names=F)


