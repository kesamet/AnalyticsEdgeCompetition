# KAGGLE COMPETITION - DEALING WITH THE TEXT DATA

# This script file is intended to help you deal with the text data provided in the competition data files

NewsTrain = read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
NewsTest = read.csv("NYTimesBlogTest.csv", stringsAsFactors=FALSE)

NewsTrain$PubDate = strptime(NewsTrain$PubDate, "%Y-%m-%d %H:%M:%S")
NewsTest$PubDate = strptime(NewsTest$PubDate, "%Y-%m-%d %H:%M:%S")
NewsTrain$Weekday = NewsTrain$PubDate$wday
NewsTest$Weekday = NewsTest$PubDate$wday
NewsTrain$Hour = NewsTrain$PubDate$hour
NewsTest$Hour = NewsTest$PubDate$hour

# Now, let's load the "tm" package

library(tm)

# Then create a corpus from the headline variable. You can use other variables in the dataset for text analytics, but we will just show you how to use this particular variable. 
# Note that we are creating a corpus out of the training and testing data.

CorpusHeadline = Corpus(VectorSource(c(NewsTrain$Headline, NewsTest$Headline)))
CorpusHeadline = tm_map(CorpusHeadline, tolower)
CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)
CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)
CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))
CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

dtm = DocumentTermMatrix(CorpusHeadline)
sparse = removeSparseTerms(dtm, 0.99)
HeadlineWords = as.data.frame(as.matrix(sparse))

# Let's make sure our variable names are okay for R:

colnames(HeadlineWords) = make.names(colnames(HeadlineWords))

# Now we need to split the observations back into the training set and testing set.
# Note that this split of HeadlineWords works to properly put the observations back into the training and testing sets, because of how we combined them together when we first made our corpus.
# Before building models, we want to add back the original variables from our datasets. 
# We'll add back the dependent variable to the training set, and the WordCount variable to both datasets. You might want to add back more variables to use in your model - we'll leave this up to you!

#HeadlineWords$WordCount = c(NewsTrain$WordCount, NewsTest$WordCount)
HeadlineWords$WordCountLog = log(c(NewsTrain$WordCount, NewsTest$WordCount)+1)
HeadlineWords$Weekday = as.factor(c(NewsTrain$Weekday, NewsTest$Weekday))
HeadlineWords$Hour = as.factor(c(NewsTrain$Hour, NewsTest$Hour))
HeadlineWords$NewsDesk = as.factor(c(NewsTrain$NewsDesk, NewsTest$NewsDesk))
HeadlineWords$SectionName = c(NewsTrain$SectionName, NewsTest$SectionName)
HeadlineWords$SubsectionName = c(NewsTrain$SubsectionName, NewsTest$SubsectionName)
for(i in 1:nrow(HeadlineWords)){
    if(nchar(HeadlineWords$SubsectionName[i])==0){
        HeadlineWords$SubsectionName[i] <- HeadlineWords$SectionName[i]
    }
}
HeadlineWords$SectionName = as.factor(HeadlineWords$SectionName)
HeadlineWords$SubsectionName = as.factor(HeadlineWords$SubsectionName)

# Now we need to split the observations back into the training set and testing set.
# Note that this split of HeadlineWords works to properly put the observations back into the training and testing sets, because of how we combined them together when we first made our corpus.
# We'll add back the dependent variable to the training set, and the WordCount variable to both datasets.
HeadlineWordsTrain = head(HeadlineWords, nrow(NewsTrain))
HeadlineWordsTest = tail(HeadlineWords, nrow(NewsTest))
HeadlineWordsTrain$Popular = as.factor(NewsTrain$Popular)


#####
#Trying new things
NewsHeadline = c(NewsTrain$Headline, NewsTest$Headline)
HeadlineWords$obama = as.factor(ifelse(grepl("obama",NewsHeadline,ignore.case=TRUE),1,0))
HeadlineWords$york = as.factor(ifelse(grepl("york",NewsHeadline,ignore.case=TRUE),1,0))
HeadlineWords$economy = as.factor(ifelse(grepl("firm|economy",NewsHeadline,ignore.case=TRUE),1,0))


#####


###################################################################
library(caret)
library(ROCR)
library(caTools)
set.seed(1)
split = sample.split(HeadlineWordsTrain$Popular, SplitRatio = 0.7)
train = subset(HeadlineWordsTrain, split==TRUE)
test = subset(HeadlineWordsTrain, split==FALSE)
test[test$SectionName=='Sports',]$SectionName = ""
test[test$SubsectionName=='Sports',]$SubsectionName = ""

table(test$Popular)
1632/nrow(test)


# CART model
library(rpart)
#library(rpart.plot)
trainCART = rpart(Popular ~ ., data=train, method="class")
predTestCART = predict(trainCART, newdata=test)[,2]
tab = table(test$Popular, predTestCART > 0.5)
(tab[1,1]+tab[2,2])/nrow(test)
performance(prediction(predTestCART, test$Popular), "auc")@y.values


# Logistic regression model
trainLog = glm(Popular ~ ., data=train, family=binomial)
predTestLog = predict(trainLog, newdata=test)
performance(prediction(predTestLog, test$Popular), "auc")@y.values #0.9246398
tab = table(test$Popular, predTestLog > 0.5)
(tab[1,1]+tab[2,2])/nrow(test)


# Random forest model
library(randomForest)
set.seed(123)
trainRF = randomForest(Popular ~ ., data=train, ntree=1000, mtry=5, importance=TRUE)
predTestRF = predict(trainRF, newdata=test, type="prob")[,2]
performance(prediction(predTestRF, test$Popular), "auc")@y.values #0.9324962
tab = table(test$Popular, predTestRF > 0.5)
(tab[1,1]+tab[2,2])/nrow(test)

nf = trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)
tr = train(Popular ~ ., data=train, method="rf", nodesize=3, ntree=3000, metric="ROC", trControl=nf)
predrf = predict(tr, newdata=test, type="prob")[,2]
performance(prediction(predrf, test$Popular), "auc")@y.values


# NN
library(nnet)
trainNN = nnet(train$Popular ~ ., train[-49], size=6, rang=0.5, decay=0.4, maxit=400)
predTestNN = predict(trainNN, newdata=test, type="raw")
performance(prediction(predTestNN, test$Popular), "auc")@y.values #0.9323869


# Combined
p = 0.98
q = 0.01
predTestComb = predTestLog*(1-p-q)+predTestRF*p+predTestNN*q
performance(prediction(predTestComb, test$Popular), "auc")@y.values #0.9376868



# GBM (not working)
gbmGrid = expand.grid(interaction.depth=13,n.trees=2300,shrinkage=0.005)
nf2 = trainControl(method="none", number=1, classProbs=TRUE, summaryFunction=twoClassSummary)
trainGBM = train(Popular ~ ., data=train, method="gbm", metric="ROC", trControl=nf2, tuneGrid=gbmGrid)
predTestGBM = plogis(predict(trainGBM, newdata=test, type="prob"))
predTestGBM = predict(trainGBM, newdata=test, type="prob")[,2]
performance(prediction(predTestGBM, test$Popular), "auc")@y.values


# GLM boost (slow)
trainGB <- train(Popular ~ ., data=train, method="glmboost")
predTestGB = predict(trainGB, newdata=test)
tab = table(test$Popular, predTestGB > 0.5)
(tab[1,1]+tab[2,2])/nrow(test)
performance(prediction(predTestGB, test$Popular), "auc")@y.values


###################################################################
# Model 1: logistic regression
HeadlineWordsLog = glm(Popular ~ ., data=HeadlineWordsTrain, family=binomial)
PredTestLog = predict(HeadlineWordsLog, newdata=HeadlineWordsTest, type="response")

# Now we can prepare our submission file for Kaggle:
MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTestLog)
write.csv(MySubmission, "SubmissionHeadlineLog2.csv", row.names=FALSE)

###################################################################
# Model 2: random forest
HeadlineWordsRF = randomForest(Popular ~ ., data=HeadlineWordsTrain, ntree=1000, mtry=5, importance=TRUE)
PredTestRF = predict(HeadlineWordsRF, newdata=HeadlineWordsTest, type="prob")[,2]

# Now we can prepare our submission file for Kaggle:
MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTestRF)
write.csv(MySubmission, "SubmissionHeadlineRF4.csv", row.names=FALSE)

###################################################################
# Model 3: NN
HeadlineWordsNN = nnet(HeadlineWordsTrain$Popular ~ ., HeadlineWordsTrain[-49], size=6, rang=0.5, decay=0.4, maxit=400)
PredTestNN = predict(HeadlineWordsNN, newdata=HeadlineWordsTest, type="raw")

# Now we can prepare our submission file for Kaggle:
MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTestNN)
write.csv(MySubmission, "SubmissionHeadlineNN.csv", row.names=FALSE)

###################################################################
# Combination
#PredTest = PredTestRF*p+PredTestLog*(1-p)
PredTest = (PredTestLog+PredTestRF+PredTestNN)/3

# Now we can prepare our submission file for Kaggle:
MySubmission = data.frame(UniqueID = NewsTest$UniqueID, Probability1 = PredTest)
write.csv(MySubmission, "SubmissionHeadlineComb3.csv", row.names=FALSE)

###################################################################
###################################################################
