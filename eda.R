library(tidyverse)
library(dplyr)
library(data.table)
library(caret)
library(xgboost)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

#### read in file and join ####
train_identity <- fread('./train_identity.csv') %>% data.frame
test_identity <- fread('./test_identity.csv') %>% data.frame

train_transaction <- fread('./train_transaction.csv') %>% data.frame
test_transaction <- fread('./test_transaction.csv') %>% data.frame


# joining transaction and identity tables (one on one but mulotiple rows for one id ?)
train <- left_join(train_transaction, train_identity)
test <- left_join(test_transaction, test_identity)

full = train %>% bind_rows(test) 
rm(train_identity,test_identity,train_transaction,test_transaction,train,test)


drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120')

full_tmp = full%>%
  mutate(istrain = ifelse(is.na(isFraud),0,1))%>%
  select(-drop_col)%>%
  mutate(
    pseudo_date = as.Date(TransactionDT%/%86400,origin = '2019-01-01'),
    pseudo_month = month(pseudo_date),
    hr = floor( (TransactionDT / 3600) %% 24 ),
    weekday = floor( (TransactionDT / 3600 / 24) %% 7)
  ) %>%
  select(-TransactionID,-TransactionDT)

full_tmp$pseudo_month%>%table()

# 
full_tmp%>%
  group_by(pseudo_month,istrain)%>%
  summarise(count_ = n())%>%
  ggplot(aes(x= as.factor(pseudo_month), y= count_,fill = istrain))+
  geom_col()

# weekday 1-->Friday has a lot more consumer 1-4 is stable saterday and sunday(the least) drop 
# but the ratio are quite the same in every day 
full_tmp%>%
  filter(!is.na(isFraud))%>%
  group_by(weekday,isFraud)%>%
  summarise(count_ = n())%>%
  ggplot(aes(x=as.factor(weekday),y=count_,fill=isFraud))+
  geom_col()



# 7 8 9 10 11 is has less Fraud ratio 
full_tmp%>%
  filter(!is.na(isFraud))%>%
  group_by(hr,isFraud)%>%
  summarise(count_ = n())%>%
  ggplot(aes(x=as.factor(hr),y=count_,fill=isFraud))+
  geom_col()

full_tmp$weekday
#### preprocess  ####



train = full%>%filter(!is.na(isFraud )) 
test =  full%>%filter(is.na(isFraud )) 

#### cut for validation 1 and self test data  from train ####
# train = 
# private test(for capping report) + 
# validation (general of capping rule )
# test for upload  (seed is use less)
index_1 = createDataPartition(y = train$TransactionDT,list = F,p = 0.8)
saveRDS(index_1,'index_1.rds')
index_1 = readRDS('index_1.rds')
train_model = train[index_1,]
train_tmp = train[-index_1,]


index_2 = createDataPartition(y = train_tmp$TransactionDT,list=F,p=0.5)
saveRDS(index_2,'index_2.rds')
index_2 = readRDS('index_2.rds')
train_cap = train_tmp[index_2,]
train_validation = train_tmp[-index_2,]
# train for model
train_model_X = train_model%>%select(-isFraud)
train_model_Y = train_model%>%select(isFraud)

# train for cap
train_cap_X = train_cap%>%select(-isFraud)
train_cap_Y = train_cap%>%select(isFraud)
xgb.cap.data = xgb.DMatrix(data.matrix(train_cap_X), missing = NA)

# train for val
train_val_X = train_val%>%select(-isFraud)
train_val_Y = train_val%>%select(isFraud)
xgb.val.data = xgb.DMatrix(data.matrix(train_val_X), missing = NA)

# test for uplopad 
test%>%head()


#### Train XGBoost ####
# find the best iteration
xgb.train.data = xgb.DMatrix(data.matrix(train_model_X), label = data.matrix(train_cap_Y), missing = NA)
param <- list(objective = "binary:logistic", base_score = 0.5)
xgboost.cv = xgb.cv(param=param, data = xgb.train.data, folds = 5, nrounds = 1500, early_stopping_rounds = 100, metrics='auc')
best_iteration = xgboost.cv$best_iteration

# trian the model using the best param
xgb.model <- xgboost(param =param,  data = xgb.train.data, nrounds=best_iteration)


# predict
xgb.preds = predict(xgb.model, xgb.cap.data)
xgb.roc_obj <- roc(test[,left], xgb.preds)
cat("XGB AUC ", auc(xgb.roc_obj))
#### Xgb importance
col_names = attr(xgb.train.data, ".Dimnames")[[2]]
imp = xgb.importance(col_names, xgb.model)
xgb.plot.importance(imp)
#### THE XGBoost Explainer
library(xgboostExplainer)
explainer = buildExplainer(xgb.model,xgb.train.data, type="binary", base_score = 0.5, trees_idx = NULL)
saveRDS(explainer, './explainer.rds')
pred.breakdown = explainPredictions(xgb.model, explainer, xgb.test.data)

cat('Breakdown Complete','\n')
weights = rowSums(pred.breakdown)
pred.xgb = 1/(1+exp(-weights))
cat(max(xgb.preds-pred.xgb),'\n')
idx_to_get = as.integer(802)
test[idx_to_get,-"left"]
showWaterfall(xgb.model, explainer, xgb.test.data, data.matrix(test[,-'left']) ,idx_to_get, type = "binary")
