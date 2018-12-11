#Clean-up work flow and optimize code
set.seed(1232018)

#create package list for project: (DOUBLE CHECK THIS)
load_pack <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}
packages <- c("ggplot2", "MASS", "DMwR", "readr", "caret", "leaps", "GGally")
load_pack(packages)

# Step 0) Load in and clean data:
nyc_hosp <- Hospital_Inpatient_Discharges_SPARCS_De_Identified_2016_csv
str(nyc_hosp)

nyc_hosp[sapply(nyc_hosp, is.character)] <- lapply(nyc_hosp[sapply(nyc_hosp_15, is.character)], 
                                                         as.factor)

nyc_hosp = rename(nyc_hosp, 
                  id = `Facility Id`,
                  Age = `Age`,
                  Sex = `Gender`,
                  race = `Race`,
                  eth = `Ethnicity`,
                  zip = `Zip Code - 3 digits`,
                  dx = `CCS Diagnosis Description`,
                  admit_type = `Type of Admission`,
                  proc = `CCS Procedure Description`,
                  reason_admit = `APR DRG Description`,
                  severity = `APR Severity of Illness Code`,
                  mort_risk = `APR Risk of Mortality`,
                  abort = `Abortion Edit Indicator`,
                  insur = `Payment Typology 1`,
                  costs = `Total Costs`,
                  
                  los = `Length of Stay`)

nyc_hosp <- nyc_hosp %>%
  filter(!is.na(los))

nyc_hosp <- na.omit(nyc_hosp)

nyc_hosp <- subset(nyc_hosp, select = c("id", "Age", "Sex", "race", "eth", 
                                        "zip", "dx", "admit_type", 
                                        "proc", "reason_admit", "severity", "mort_risk", "abort", "insur", "costs", "los"))


train.nyc <- nyc_hosp[0:125239, ] #for an 80/20 split
test.nyc <- nyc_hosp[125240:166986,]

####################### Step 1) EDA
hist(nyc_hosp$`Length of Stay`)
hist(nyc_hosp$`Total Charges`)
names(nyc_hosp)
plot(x=nyc_hosp$race, y=nyc_hosp$costs)
plot(x=nyc_hosp$mort_risk, y=nyc_hosp$costs)
plot(x=nyc_hosp$Age, y=nyc_hosp$costs)
plot(x=nyc_hosp$los, y=nyc_hosp$costs)

#LOS could be more normalized with a log transformation
log_los <- log(los)
nyc_hosp <- cbind(log_los, nyc_hosp)

cor(nyc_hosp$log_los, nyc_hosp$costs)

#could mortality be modeled?
ggplot(nyc_hosp, aes(mort_risk)) + 
  geom_bar()
ggplot(nyc_hosp, aes(mort_risk, fill = race)) +
  geom_bar()
ggplot(nyc_hosp, aes(mort_risk, fill = Sex)) +
  geom_bar()

#log of length of stay
ggplot(nyc_hosp, aes(log_los, fill = race)) + 
  geom_density(alpha=.5)

#cost by race
ggplot(nyc_hosp, aes(costs, fill = race)) +
  geom_density(alpha = .5, adjust = 1/5)


# start exploring with different demographic and in-hospital data
summary(lm(log_los ~ costs, data = nyc_hosp))
summary(lm(log_los ~ costs + Sex + insur, data = nyc_hosp)) ######highest R square, moderate correlation with costs 

#exploring other models
summary(lm(costs ~ mort_risk, data=nyc_hosp))
summary(lm(cost_charge_ratio ~ race, data = nyc_hosp))
summary(glm(mort_risk ~ race, data = nyc_hosp))
lm.log_los <- lm(log_los ~ race + Sex + insur + admit_type + severity + mort_risk, data=nyc_hosp)
summary(lm.log_los)


####################Step 2) We will fit a linear model and a penalized linear model. Build training models:
set.seed(1232018)
train.control <- trainControl(method = "cv", number = 10)
pls.mod.train <- train(log_los ~.-los -cost_charge_ratio, method = "pls", data = train.nyc,
                       tuneLength = 20,
                       trControl = train.control,
                       preProc = c("zv","center","scale"))
set.seed(1232018)
lm.mod.train <- train(log_los ~.-los -cost_charge_ratio, method = 'lm', data= train.nyc,
                      tuneLength = 20,
                      trControl = train.control,
                      preProc = c("zv","center","scale"))
par(mfrow=c(2,2))
plot(pls.mod.train)
plot(lm.mod.train)
summary(pls.mod.train)
summary(lm.mod.train)

model_perf <- (resamples(list("PLS" = pls.mod.train, "LM" = lm.mod.train)))
rmse_mod_plot <- bwplot(model_perf, metric = "RMSE")
rsquare_mod_blot <- bwplot(model_perf, metric = "Rsquared")

plot(varImp(pls.mod.train), 10, main = "Penalized Linear Model")
plot(varImp(lm.mod.train), 10, main = "Linear Model")

#plot residuals of each model 
ggplot(data=train.nyc, aes(lm.mod.train$residuals)) + 
  geom_histogram(binwidth = 1, color = "black", fill = "purple4") +
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x=element_line(),
        axis.line.y=element_line()) +
  ggtitle("Histogram for Model Residuals") #residuals, while overall high, are centered around 0

#################Step 3) Make predictions with model on test dataset
require(dplyr)
require(DMwR)
test.nyc_16 <- as.tibble(test.nyc)
test.nyc_16 %>% dplyr::select(-1)

set.seed(1232018)
pred_lm_16 <- predict(lm.mod.train, test.nyc_16)
pred_pls_16 <- predict(pls.mod.train, test.nyc_16)

actuals_preds_lm_16 <- data.frame(cbind(actuals=test.nyc$log_los, predicteds=pred_lm_16))
corr_acc_lm_16 <- cor(actuals_preds_lm_16) #presents the comparison of predicted to actual vals of log(LOS)

actuals_preds_pls_16 <- data.frame(cbind(actuals=test.nyc$log_los, predicteds=pred_pls_16))
corr_acc_pls_16 <- cor(actuals_preds_pls_16)

lm_mod16_eval <- DMwR::regr.eval(actuals_preds_lm_16$actuals, actuals_preds_lm_16$predicteds) #uses package to evaluate predictive capability of the model
pls_mod16_eval <- DMwR::regr.eval(actuals_preds_pls_16$actuals, actuals_preds_pls_16$predicteds)
