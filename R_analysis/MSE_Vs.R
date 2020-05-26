library(dplyr)
library(readr)
library(ggplot2)
library(truncnorm)

#function for cleaning up Vs
clean_V <- function(column){
  column <- column %>%
    lapply(function(x){gsub(pattern = "[", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = "]", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = ";", replacement="",x, fixed = TRUE)})
}

dealing_with_frequency_tables_Vs <- function(ft){
  ft <- ft  %>% lapply(function(x){gsub(pattern = "Dict(\"Array{String,N} where N", replacement="",x, fixed = TRUE)})
  ft <- ft  %>% lapply(function(x){gsub(pattern = "Dict(Array{String,N} where N", replacement="",x, fixed = TRUE)})
  
  n_objects <- 5
  frequency_table_as_list <- as.list(strsplit(ft[[1]], ",", fixed=TRUE)[[1]])
  len_ft <- length(frequency_table_as_list)
  
  weights <- vector(mode="double", length=len_ft) #will hold number of times each different reality was sampled in particle filter
  Vs <- lapply(1:len_ft, function(x) matrix(NA, nrow=n_objects, ncol=2))
   
  for(j in 1:len_ft){
    string <- frequency_table_as_list[[j]]
    start_for_weights <- regexpr("=>", string)
    stop_for_weights <- nchar(string)
    weights_as_str <- substring(string, start_for_weights, stop_for_weights)
    matches <- regmatches(weights_as_str, gregexpr("[[:digit:]]+", weights_as_str))
    weights[j] <- as.numeric(unlist(matches))
    
    V_as_str <- substring(string, 1, start_for_weights)
    matches <- regmatches(V_as_str, gregexpr("[[:digit:]].[[:digit:]]+", V_as_str)) #for 0.blah
    V <- matrix(as.numeric(unlist(matches)), ncol=2, byrow=TRUE)
    Vs[[j]] <- V
  }
  return(list("Vs_as_list" = Vs, "weights" = weights))
}

MSE_Vs <- function(data){
  
  #clean up the data
  data$gt_R <- data$gt_R %>%
    lapply(function(x){gsub(pattern = "Any", replacement="",x, fixed = TRUE)})
  
  list <- grep('frequency.table.Vs.after.p', colnames(data), value=TRUE)
  n_percepts = length(list)
  #n_objects (length of possible objects)
  n_objects = 5
  
  mode_Vs <- lapply(1:n_percepts, function(x) matrix(NA, nrow=n_objects, ncol=2))
  for(p in 1:n_percepts){
    returned <- dealing_with_frequency_tables_Vs(data[[list[p]]])
    Vs_as_list <- returned$Vs_as_list
    weights <- returned$weights
    index <- which(weights==max(weights))[1] #take the first one in case of tie
    mode_Vs[p] <- Vs_as_list[index]
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
  
  MSE_exp_FA = sum((gt_V[,1] - exp_mat[,1])^2)/n_objects
  MSE_exp_M = sum((gt_V[,2] - exp_mat[,2])^2)/n_objects

  
  MSE_FA <- vector(mode="double", length=n_percepts)
  MSE_M <- vector(mode="double", length=n_percepts)
  for(p in 1:n_percepts){
    #temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1] #[-1] is needed sometimes because there might be a space at the beginning of gt_V
    #temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))
    #mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
    mat <- mode_Vs[[p]]
    MSE_FA[p] <- sum((gt_V[,1] - mat[,1])^2)/n_objects
    MSE_M[p] <- sum((gt_V[,2] - mat[,2])^2)/n_objects
  }
  
  percept_number <- rep(1:n_percepts)
  exp_MSE_FA <- rep(MSE_exp_FA, n_percepts)
  exp_MSE_M <- rep(MSE_exp_M, n_percepts)
  
  gt_FA_airplane <- rep(gt_V[5,1], n_percepts)
  gt_M_airplane <- rep(gt_V[5,2], n_percepts)
  
  toPlot <- data.frame(percept_number, exp_MSE_FA, exp_MSE_M, MSE_FA, MSE_M, gt_FA_airplane, gt_M_airplane)

  return(toPlot)
}

