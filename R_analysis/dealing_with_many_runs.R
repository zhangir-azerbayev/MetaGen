library(dplyr)
library(tidyr)
library(readr)
library(tidyboot)
library(ggplot2)
library(tidyverse)
library(Rfast)

source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy.R")
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/visualize_reality.R")
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/MSE_Vs.R")
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/parse_ambiguous_percept.R")
#source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy_fake.R")

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

# raw_data <- read_delim("merged.csv",
#                         "&", escape_double = FALSE, trim_ws = TRUE, n_max = 500)

raw_data <- read_delim("merge_23.csv",
                        "&", escape_double = FALSE, trim_ws = TRUE, n_max = 500)


# raw_data <- read_delim("merge_the_merges.csv",
#                        "&", escape_double = FALSE, trim_ws = TRUE)
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
# raw_data <- na.omit(raw_data)

# raw_data <- raw_data %>% 
#   mutate_all(~ifelse(. %in% c("N/A", "null", ""), NA, .)) %>% 
#   na.omit()
# 
# raw_data.dropna(axis = 0, how = 'any')

#some files were messed up. seems that sometime the header gets read into
#the value for a column. Want to remove rows where this happened. Could
#have been the result of a cancelation on the cluster
data2 <- raw_data
data2 <- data2 %>% filter_all(all_vars(.!= "gt_R"))

I <- nrow(data2)
#I <- 3477 #3477 for partially completed run
#I <- 10

#combined_data <- map_df(1:I,function(x){return(cbind(simID = x, MSE_Vs(raw_data[x,]), accuracy(raw_data[x,]), inferred_ambiguous_percept = parse_ambiguous_percept(raw_data[x,])))})
#combined_data <- map_df(1:I,function(x){return(cbind(simID = x, MSE_Vs(raw_data[x,]), accuracy(raw_data[x,])))})

#combined_data <- map_df(1:I,function(x){return(cbind(simID = x, accuracy_fake(raw_data[x,])))})
#I <- 500
combined_data <- vector(mode = "list", length = length(data))
for(i in 1:I){
  print(i)
  combined_data[[i]] <- cbind(simID = i, MSE_Vs(data2[i,]), accuracy(data2[i,]))
}
combined_data <- combined_data %>% bind_rows


###################################################################

#find noisiest percept
noisiest_row <- combined_data[which(combined_data$perceived_noise == max(combined_data$perceived_noise)),]
noisiest_sim <- noisiest_row$simID
noisiest_percept <- noisiest_row$percept_number

noisiest_df <- raw_data %>% filter(simID == noisiest_sim[1])
accuracy(noisiest_df)$perceived_noise #sanity check fails
visualize_reality(noisiest_df) #have to uncomment each graph that I want to see. or put a break point
#another way
data <- noisiest_df #then run analysis_of_Vs line by line

#where thresholding gets it wrong
which(combined_data$A_threshold==0)
#end noisiest percept

###################################################################
accuracy_plot <- function(data){
  #drop rows for percept0
  data <- na.omit(data)
  
  df_Accuracy <-
    data %>% gather(
      Model,
      Score,
      A_retrospective_metagen,
      A_lesioned_metagen,
      A_online_metagen,
      #A_naive_reality,
      A_threshold
    )
  
  GetLowerCI <- function(x,y){return(prop.test(x,y)$conf.int[1])}
  GetTopCI <- function(x,y){return(prop.test(x,y)$conf.int[2])}
  
  toPlot_Accuracy <- df_Accuracy %>% group_by(percept_number,Model) %>% summarize(Samples=n(),Hits=sum(Score),Mean=mean(Score),Lower=GetLowerCI(Hits,Samples),Top=GetTopCI(Hits,Samples))
  
  #group_by(percept_number)
  #MyDataFrame <- df %>% group_by(percept_number, Model) %>% tidyboot_mean(Score)
  
  ggplot(
    toPlot_Accuracy,
    aes(
      x = percept_number,
      y = Mean,
      ymin = Lower,
      ymax = Top,
      fill = Model,
      group = Model
    )
  ) + geom_ribbon() + geom_line() + coord_cartesian(ylim = c(0.70, 1)) + theme(aspect.ratio=1)
}


###################################################################
mse_V_plot <- function(data){
  df_V <-
    data %>% gather(
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
  
  toPlot_V <- df_V %>% group_by(percept_number,V_param) %>% summarize(Mean_MSE=GetMean(MSE),Lower=GetLowerCI(MSE),Top=GetTopCI(MSE))
  
  ggplot(
    toPlot_V,
    aes(
      x = percept_number,
      y = Mean_MSE,
      ymin = Lower,
      ymax = Top,
      fill = V_param,
      group = V_param
    )
  ) + geom_ribbon() + geom_line() + coord_cartesian(ylim = c(0, 0.02)) + theme(aspect.ratio=1)
}

###################################################################
noise_vs_accuracy_plot <- function(data){
  #drop rows for percept0
  data <- na.omit(data)
  
  #fail <- data %>% filter(perceived_noise<0.05 & A_retrospective_metagen == 0)
  
  df_Accuracy <-
    data %>% gather(
      Model,
      Score,
      A_retrospective_metagen,
      #A_lesioned_metagen,
      #A_online_metagen,
      #A_naive_reality,
      A_threshold
    )
  
  #GetLowerCI <- function(x,y){return(prop.test(x,y)$conf.int[1])}
  #GetTopCI <- function(x,y){return(prop.test(x,y)$conf.int[2])}
  
  toPlot_Accuracy <- df_Accuracy %>% group_by(Model)# %>% summarize(Samples=n(),Hits=sum(Score),Mean=mean(Score),Lower=GetLowerCI(Hits,Samples),Top=GetTopCI(Hits,Samples))
  
  #fail <- toPlot_Accuracy %>% filter(perceived_noise<0.05 & A_retrospective_metagen == 0)
  #fail <- toPlot_Accuracy %>% filter(perceived_noise<0.05 & Score == 0 & Model=="A_retrospective_metagen")
  
  b <- ggplot(toPlot_Accuracy, aes(x = perceived_noise, y = Score))
  b + 
    #geom_point(aes(color = Model)) +
    scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
    #ggline(lowess(Score ~ perceived_noise, toPlot_Accuracy))
    geom_smooth(aes(color = Model), method="gam") + theme(aspect.ratio=1)
  
  # #Warning message:
  # Computation failed in `stat_smooth()`:
  #   workspace required (8438306300) is too large probably because of setting 'se = TRUE'.
}


###################################################################
noise_vs_differences <- function(data){
  #drop rows for percept0
  data <- na.omit(data)

  b <- ggplot(data, aes(x = perceived_noise, y = diff_between_retro_and_threshold))
  b + 
    #geom_point() +
    scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
    #ggline(lowess(Score ~ perceived_noise, toPlot_Accuracy))
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_smooth(method="gam")
}

###################################################################
noise_density_plot <- function(data){
  #drop rows for percept0
  data <- na.omit(data)
  
  p <- ggplot(data, aes(x=perceived_noise)) + 
    geom_density()
  p
}

###################################################################
accuracy_plot(combined_data)
mse_V_plot(combined_data)
noise_vs_accuracy_plot(combined_data)
noise_vs_differences(combined_data)
noise_density_plot(combined_data)

#Cherry-pick run to visualize...
#Pick sim with biggest avg difference between retro and threshold models
m <- max(abs(combined_data$avg_diff_between_retro_and_threshold), na.rm = TRUE)
biggest_win_for_metagen <- combined_data %>% filter(avg_diff_between_retro_and_threshold==m)
sim <- biggest_win_for_metagen$simID[1]
visualize_reality(raw_data[sim,])

data <- raw_data[sim,]

subset <- combined_data[1:500,]
