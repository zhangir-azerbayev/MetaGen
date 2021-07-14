#This is file provides support functions called by the main analysis_of_raw_data.R script

library(tidyverse)

#function for cleaning up Vs
clean_V <- function(column){
  column <- column %>%
    lapply(function(x){gsub(pattern = "[", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = "]", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = ";", replacement="",x, fixed = TRUE)})
}

MSE_Vs <- function(data){
  
  list <- grep('avg V', colnames(data), value=TRUE)
  n_percepts = length(list)
  #n_objects (length of possible objects)
  n_objects = 5
  
  for(p in 1:n_percepts){
    data[[list[p]]] <- clean_V(data[[list[p]]])
  }
  
  data$gt_V <- clean_V(data$`gt_V `)

  #look at how V changes after each percept for the PFs
  temp_var <- unlist(strsplit(as.character(data$gt_V), split = " "))
  gt_V <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
  
  M_mean = 0.5 #mean of a beta with alpha = 2 and beta = 10
  FA_mean = 0.5
  exp_mat <- cbind(rep(FA_mean, n_objects), rep(M_mean, n_objects)) #for when it's fa rather than hall lambda
  
  MSE_exp_FA = sum((gt_V[,1] - exp_mat[,1])^2)/n_objects
  MSE_exp_M = sum((gt_V[,2] - exp_mat[,2])^2)/n_objects

  
  MSE_FA <- vector(mode="double", length=n_percepts)
  MSE_M <- vector(mode="double", length=n_percepts)
  for(p in 1:n_percepts){
    #temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1] #[-1] is needed sometimes because there might be a space at the beginning of gt_V
    temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1]
    mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
    MSE_FA[p] <- sum((gt_V[,1] - mat[,1])^2)/n_objects
    MSE_M[p] <- sum((gt_V[,2] - mat[,2])^2)/n_objects
  }
  
  percept_number <- rep(1:n_percepts)
  exp_MSE_FA <- rep(MSE_exp_FA, n_percepts)
  exp_MSE_M <- rep(MSE_exp_M, n_percepts)
  
  toPlot <- data.frame(percept_number, exp_MSE_FA, exp_MSE_M, MSE_FA, MSE_M)

  return(toPlot)
}

mse_V_plot <- function(toPlot){

  #GetMean <- function(x){return(t.test(x)$estimate)}
  #GetLowerCI <- function(x){return(t.test(x)$conf.int[1])}
  #GetTopCI <- function(x){return(t.test(x)$conf.int[2])}
  
  df_FA = gather(toPlot, "FA_line", "value", c(exp_MSE_FA, MSE_FA)) %>%
    group_by(FA_line)
  
  p <- ggplot(
    df_FA,
    aes(
      x = percept_number,
      y = value
    )
  ) + geom_line(aes(color = FA_line)) + theme(aspect.ratio=1)
  
  ggsave("mse_FA_plot.pdf",p)
  
  df_M = gather(toPlot, "M_line", "value", c(exp_MSE_M, MSE_M)) %>%
    group_by(M_line)
  
  q <- ggplot(
    df_M,
    aes(
      x = percept_number,
      y = value
    )
  ) + geom_line(aes(color = M_line)) + theme(aspect.ratio=1)
  
  ggsave("mse_M_plot.pdf", q)
}
