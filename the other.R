library(readr)
library(tidyverse)
library(MLmetrics)
library(lightgbm)
options(scipen = 99)

train_iden <- read_csv("../input/train_identity.csv")
train_trans <- read_csv("../input/train_transaction.csv")
test_iden <- read_csv("../input/test_identity.csv")
test_trans <- read_csv("../input/test_transaction.csv")

y <- train_trans$isFraud 
train_trans$isFraud <- NULL

drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120')

train <- train_trans %>% left_join(train_iden)
test <- test_trans %>% left_join(test_iden)
rm(train_iden,train_trans,test_iden,test_trans) ; invisible(gc())

# using single hold-out validation (20%)
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT,0.8))
train[,drop_col] <- NULL
test[,drop_col] <- NULL


tem <- train %>% bind_rows(test) %>%
  mutate(hr = floor( (TransactionDT / 3600) %% 24 ),
         weekday = floor( (TransactionDT / 3600 / 24) %% 7)
  ) %>%
  select(-TransactionID,-TransactionDT)

#############################################################################################################
# FE part1 : Count encoding
char_features <- tem[,colnames(tem) %in% 
                       c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                         "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                         "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                         "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                         "id_37","id_38")]

fe_part1 <- data.frame(0)
for(a in colnames(char_features) ){
  tem1 <- char_features %>% group_by(.dots = a) %>% mutate(count = length(card4)) %>% ungroup() %>% select(count)
  colnames(tem1) <- paste(a,"__count_encoding",sep="")
  fe_part1 <- data.frame(fe_part1,tem1)
}

fe_part1 <- fe_part1[,-1]
rm(char_features,tem1) ; invisible(gc())
cat("fe_part1 ncol :" , ncol(fe_part1) ,"\n" )
#############################################################################################################
# label 

char_features <- colnames(tem[, sapply(tem, class) %in% c('character', 'factor')])
for (f in char_features){
  levels <- unique(tem[[f]])
  tem[[f]] <- as.integer(factor(tem[[f]], levels=levels))
}


tem <- data.frame(tem,fe_part1)

train <- tem[1:nrow(train),]
test <- tem[-c(1:nrow(train)),]
rm(tem) ; invisible(gc())

############################################################################################################
# model

cat("train_col :" , ncol(train), "test_col :", ncol(test) ,"\n" )


d0 <- lgb.Dataset(data.matrix( train[tr_idx,] ), label = y[tr_idx] )
dval <- lgb.Dataset(data.matrix( train[-tr_idx,] ), label = y[-tr_idx] ) 

# not tuned

lgb_param <- list(boosting_type = 'gbdt',
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = "false",
                  learning_rate = 0.006883242363721497,
                  num_leaves = 491,
                  max_depth = -1,
                  bagging_seed = 11,
                  min_child_weight = 0.03454472573214212,
                  feature_fraction = 0.3797454081646243,
                  bagging_freq = 1,
                  bagging_fraction =  0.4181193142567742,
                  min_data_in_leaf = 106,
                  reg_alpha = 0.3899927210061127,
                  reg_lambda = 0.6485237330340494,
                  random_state = 47
)


valids <- list(valid = dval)
lgb <- lgb.train(params = lgb_param,  data = d0, nrounds = 15000, 
                 eval_freq = 200, valids = valids, early_stopping_rounds = 200, verbose = 1)


oof_pred <- predict(lgb, data.matrix(train[-tr_idx,]))
cat("best iter :" , lgb$best_iter, "best score :", AUC(oof_pred, y[-tr_idx]) ,"\n" )
iter <- lgb$best_iter

rm(lgb,d0,dval) ; invisible(gc())

# full data
d0 <- lgb.Dataset( data.matrix( train ), label = y )
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter*1.05, verbose = -1)
pred <- predict(lgb, data.matrix(test))

imp <- lgb.importance(lgb)
sub <- data.frame(read_csv("../input/sample_submission.csv"))
sub[,2] <- pred

write.csv(sub,"sub.csv",row.names=F)
write.csv(imp,"imp.csv",row.names=F)