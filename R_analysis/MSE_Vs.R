library(dplyr)
library(readr)
library(ggplot2)
library(truncnorm)

MSE_Vs <- function(data){
  
  #clean up the data
  data$gt_R <- data$gt_R %>%
    lapply(function(x){gsub(pattern = "Any", replacement="",x, fixed = TRUE)})
  
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
  
  data$gt_V <- clean_V(data$gt_V)
  
  #analysis questions
  
  
  #look at how V changes after each percept for the PFs
  #avg.V.after.p1
  
  temp_var <- unlist(strsplit(as.character(data$gt_V), split = " "))
  gt_V <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
  
  beta_mean = 2/(2+10) #mean of a beta with alpha = 2 and beta = 10
  temp_var <- rep(beta_mean, n_objects)
  exp_mat <- cbind(temp_var, temp_var) #for when it's fa rather than hall lambda
  
  MSE_exp_FA = sum((gt_V[,1] - exp_mat[,1])^2)/5
  MSE_exp_M = sum((gt_V[,2] - exp_mat[,2])^2)/5
  
  
  #n_objects (length of possible objects)
  n_objects = 5
  
  MSE_FA <- vector(mode="double", length=n_percepts)
  MSE_M <- vector(mode="double", length=n_percepts)
  for(p in 1:n_percepts){
    temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1] #[-1] is needed sometimes because there might be a space at the beginning of gt_V
    #temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))
    mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
    MSE_FA[p] <- mean((gt_V[,1] - mat[,1])^2)
    MSE_M[p] <- mean((gt_V[,2] - mat[,2])^2)
  }
  
  percept_number <- rep(1:n_percepts)
  exp_MSE_FA <- rep(MSE_exp_FA, n_percepts)
  exp_MSE_M <- rep(MSE_exp_M, n_percepts)
  toPlot <- data.frame(percept_number, exp_MSE_FA, exp_MSE_M, MSE_FA, MSE_M)

  return(toPlot)
}

