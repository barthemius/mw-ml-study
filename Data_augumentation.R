library(rio)
library(UBL)
library(tidyverse)
library(plotly)
library(ggplot2)
library(reshape2)

data_raw <- import("patient_data_all_check_V11.csv")

# Replace kobieta mężczynza
km <- function(x) {
  if (x == "Kobieta")
  {
    0
  }
  else
    1
}

data_raw['sex'] <- apply(data_raw['sex'], 1, km)
data_raw['Smoke'] <- apply(data_raw['Smoke'], 1, function(x) {ifelse(x == "Tak", 1, 0)})
data_raw['Smoke_Years'] <- apply(data_raw['Smoke_Years'], 1, function(x) {ifelse(is.na(x), 0, x)})
data_raw['Smoke_amount_day'] <- apply(data_raw['Smoke_amount_day'], 1, function(x) {ifelse(is.na(x), 0, x)})

data_raw <- subset(data_raw, select = -c(id, Likelihood_of_obesity, Likelihood_of_coronary_heart_disease))
head(data_raw)

data_augumented <- SMOGNRegress(Likelihood_of_diabetes ~.,
                                data_raw,
                                dist = "HEOM",
                                C.perc = "balance",
                                thr.rel = 0.8
                                )


hist(data_raw$Likelihood_of_diabetes)
hist(data_augumented$Likelihood_of_diabetes)

gg <- ggplot(data_augumented,aes(x = Likelihood_of_diabetes, color = 'density')) +  
  geom_histogram(aes(y = ..density..), bins = 20,  fill = '#67B7D1', alpha = 0.5) +  
  geom_density(color = '#67B7D1') +  
  geom_rug(color = '#67B7D1') + 
  ylab("") + 
  xlab("")  + theme(legend.title=element_blank()) +
  scale_color_manual(values = c('density' = '#67B7D1'))
gg

#ggplot(data_raw, aes(x = Likelihood_of_diabetes)) + geom_histogram(aes(y = ..density..), bins = 20)

# Export data to csv
write_csv(data_augumented, "patient_data_diab_augumented.csv")
