#This is analysis for with_fragmentation

library(dplyr)
library(readr)
library(ggplot2)
library(truncnorm)

#data <- list.files("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/Simulated_data_and_analysis/with10realities/test", pattern='output') %>%
  #lapply(function(x){read.csv(x, header=TRUE, sep='&')})
#  lapply(function(x) {
#    tmp <- try(read.table(x, header = TRUE, sep = '&'))
#    if (!inherits(tmp, 'try-error')) tmp
#  })%>%
#  bind_rows

data <- read.csv("output111.csv",header=TRUE, sep='&')

#clean up the data
data$mode.realities.PF <- data$mode.realities.PF  %>%
  lapply(function(x){gsub(pattern = "Array{String,N} where N", replacement="",x, fixed = TRUE)})

#function for cleaning up Vs
clean_V <- function(column){
  column <- column %>%
    lapply(function(x){gsub(pattern = "[", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = "]", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = ";", replacement="",x, fixed = TRUE)})
}

list <- grep('avg.V.after.p', colnames(data), value=TRUE)
n_percepts = length(list)
for(p in 1:n_percepts){
  data[[list[p]]] <- clean_V(data[[list[p]]])
}

#analysis questions


#look at how V changes after each percept for the PFs
#avg.V.after.p1

#n_objects (length of possible objects)
n_objects = 5

#truncated normal prior for hallucination lambda
trunc_norm_mean <- qtruncnorm(0.5, 0, 100, 0.2, 0.5)
temp_var1 <- rep(trunc_norm_mean, n_objects)
#beta distribution prior for miss rate
beta_mean = qbeta(0.5, 2, 10)
temp_var2 <- rep(beta_mean, n_objects)

exp_mat <- cbind(temp_var1, temp_var2)


MSE_hall <- vector(mode="double", length=n_percepts)
MSE_M <- vector(mode="double", length=n_percepts)
for(p in 1:n_percepts){
  temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1]
  mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
  MSE_hall[p] <- mean((gt_V[,1] - mat[,1])^2)
  MSE_M[p] <- mean((gt_V[,2] - mat[,2])^2)
}
x <- rep(1:n_percepts)


category_names = c("person","bicycle","car","motorcycle","airplane")

num_categories = length(category_names)
for(cat in 1:num_categories){
  
  cat = 2
  
  MSE_beta_mean_FA_cat = (gt_V[cat,1] - exp_mat[cat,1])^2
  MSE_beta_mean_M_cat = (gt_V[cat,2] - exp_mat[cat,2])^2
  
  MSE_FA_cat <- vector(mode="double", length=n_percepts)
  MSE_M_cat <- vector(mode="double", length=n_percepts)
  FA_cat <- vector(mode="double", length=n_percepts)
  M_cat <- vector(mode="double", length=n_percepts)
  
  list <- grep('avg.V.after.p', colnames(data), value=TRUE)
  for(p in 1:n_percepts){
    temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1]
    mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
    FA_cat[p] <- mat[cat,1]
    M_cat[p] <- mat[cat,2]
    MSE_FA_cat[p] <- mean((gt_V[cat,1] - mat[cat,1])^2)
    MSE_M_cat[p] <- mean((gt_V[cat,2] - mat[cat,2])^2)
  }
  x <- rep(1:n_percepts)
  
  #Misses and FAs
  
  gt_R_as_list <- strsplit(as.character(data$gt_R), split = "\\], \\[")
  gt_R_as_list <- gt_R_as_list[[1]]
  
  
  miss_count <- vector("integer", length=n_percepts)
  false_alarm_count <- vector("integer", length=n_percepts)
  
  list <- grep('percept', colnames(data), value=TRUE)
  for(index in 1:length(gt_R_as_list)){
    
    P_as_list <- strsplit(as.character(data[[list[index]]]), split = "\\]")
    P_as_list <- P_as_list[[1]][-11]#getting rid of empty quotes at end.
    n_frames <- length(P_as_list)
    
    #if it's in the reality
    if(grepl(category_names[cat], gt_R_as_list[index])){
      
      #if it's not in the frame
      for(f in 1:n_frames){
        if(grepl(category_names[cat], P_as_list[f])!=TRUE){
          miss_count[index] = miss_count[index] + 1
        }
      }
    } else {
      #if it's in the frame
      for(f in 1:n_frames){
        if(grepl(category_names[cat], P_as_list[f])){
          false_alarm_count[index] = false_alarm_count[index] + 1
        }
      }
    }
  }
  
  toPlot <- data.frame(x, FA_cat, M_cat, MSE_FA_cat, MSE_M_cat, false_alarm_count, miss_count)
  
  ggplot(toPlot, aes(x=x, y=M_cat, color=miss_count )) + 
    geom_point() +
    geom_hline(yintercept = gt_V[cat,2]) +
    ggtitle(category_names[cat]) +
    coord_cartesian(ylim = c(0, 1)) 
  
  # ggplot(toPlot, aes(x=x, y=MSE_M_cat, color=miss_count )) + 
  #   geom_point() +
  #   geom_hline(yintercept = MSE_beta_mean_M_cat) +
  #   ggtitle(category_names[cat])
  
  ggplot(toPlot, aes(x=x, y=FA_cat, color=false_alarm_count )) + 
    geom_point() +
    geom_hline(yintercept = gt_V[cat,1]) +
    ggtitle(category_names[cat]) +
    coord_cartesian(ylim = c(0, 1)) 
  
  # ggplot(toPlot, aes(x=x, y=MSE_FA_cat, color=false_alarm_count )) + 
  #   geom_point() +
  #   geom_hline(yintercept = MSE_beta_mean_FA_cat) +
  #   ggtitle(category_names[cat])
  
}


