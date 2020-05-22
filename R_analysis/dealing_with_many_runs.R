library(dplyr)
library(tidyr)
library(readr)
library(tidyboot)
library(ggplot2)
library(tidyverse)

source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy.R")
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/visualize_reality.R")
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/MSE_Vs.R")

# data <-
#   list.files(
#     "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/Simulated_data_and_analysis/May/test",
#     pattern = 'output'
#   ) %>%
#   lapply(function(x) {
#     read.csv(x, header = TRUE, sep = '&')
#   })
##data = data %>% bind_rows

# combined_data <- vector(mode = "list", length = length(data))
# for(i in 1:length(data)){
#   combined_data[[i]] <- cbind(simID = i, accuracy(data[[i]]))
# }
# combined_data <- combined_data %>% bind_rows

# raw_data <- read_delim("output111.csv",
#                        "&", escape_double = FALSE, trim_ws = TRUE)

raw_data <- read_delim("merged.csv",
                        "&", escape_double = FALSE, trim_ws = TRUE)
names(raw_data)<-str_replace_all(names(raw_data), c(" " = "."))

simID <- 1:dim(raw_data)[1]
raw_data <- cbind(simID, raw_data)

##################################

# data[1,] %>% accuracy()
# apply(data, 1, FUN=accuracy(x))
# 
# accuracy(data[1,])

#Row 3478 is messed up. I think these didn't finish running or something
#and row 3510
# data <- data[-3478,]
raw_data <- na.omit(raw_data)

#I <- nrow(raw_data)
I <- 1000 #3477 for partially completed run

combined_data <- map_df(1:I,function(x){return(cbind(simID = x, MSE_Vs(raw_data[x,]), accuracy(raw_data[x,])))})

###################################################################

#find noisiest percept
noisiest_row <- combined_data[which(combined_data$perceived_noise == max(combined_data$perceived_noise)),]
noisiest_sim <- noisiest_row$simID
noisiest_percept <- noisiest_row$percept_number

noisiest_df <- raw_data %>% filter(simID == noisiest_sim)
visualize_reality(noisiest_df) #have to uncomment each graph that I want to see. or put a break point
#another way
data <- noisiest_df #then run analysis_of_Vs line by line

#where thresholding gets it wrong
which(combined_data$A_threshold==0)
#end noisiest percept

###################################################################

df <-
  combined_data %>% gather(
    Model,
    Score,
    A_retrospective_metagen,
    A_lesioned_metagen,
    A_online_metagen,
    A_naive_reality,
    A_threshold
  )

GetLowerCI <- function(x,y){return(prop.test(x,y)$conf.int[1])}
GetTopCI <- function(x,y){return(prop.test(x,y)$conf.int[2])}

MyDataFrame <- df %>% group_by(percept_number,Model) %>% summarize(Samples=n(),Hits=sum(Score),Mean=mean(Score),Lower=GetLowerCI(Hits,Samples),Top=GetTopCI(Hits,Samples))

#group_by(percept_number)
#MyDataFrame <- df %>% group_by(percept_number, Model) %>% tidyboot_mean(Score)

ggplot(
  MyDataFrame,
  aes(
    x = percept_number,
    y = Mean,
    ymin = Lower,
    ymax = Top,
    fill = Model,
    group = Model
  )
) + geom_ribbon() + geom_line()

###################################################################

df <-
  combined_data %>% gather(
    V_param,
    MSE,
    MSE_FA,
    MSE_M,
    exp_MSE_FA,
    exp_MSE_M,
  )

GetMean <- function(x){return(t.test(x)$estimate)}
GetLowerCI <- function(x){return(t.test(x)$conf.int[1])}
GetTopCI <- function(x){return(t.test(x)$conf.int[2])}

MyDataFrame <- df %>% group_by(percept_number,V_param) %>% summarize(Mean_MSE=GetMean(MSE),Lower=GetLowerCI(MSE),Top=GetTopCI(MSE))

ggplot(
  MyDataFrame,
  aes(
    x = percept_number,
    y = Mean_MSE,
    ymin = Lower,
    ymax = Top,
    fill = V_param,
    group = V_param
  )
) + geom_ribbon() + geom_line() + coord_cartesian(ylim = c(0, 0.1)) + coord_cartesian(ylim = c(0, 0.1))
