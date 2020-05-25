library(dplyr)
library(readr)
library(ggplot2)
library(truncnorm)
library(stringr)
#library(assertthat)
library(cowplot)

#function for cleaning up Vs
clean <- function(column){
  column <- column %>%
    # lapply(function(x){gsub(pattern = "[", replacement="",x, fixed = TRUE)}) %>%
    # lapply(function(x){gsub(pattern = "]", replacement="",x, fixed = TRUE)}) %>%
    # lapply(function(x){gsub(pattern = ";", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = "Any", replacement="",x, fixed = TRUE)})
  # lapply(function(x){gsub(pattern = "\\", replacement="",x, fixed = TRUE)})
}

dealing_with_frequency_tables <- function(ft, n_percepts){
  ft <- ft  %>% lapply(function(x){gsub(pattern = "Dict(\"Array{String,N} where N", replacement="",x, fixed = TRUE)})
  
  frequency_table_as_list <- as.list(strsplit(ft[[1]], "Array{String,N} where N", fixed=TRUE)[[1]])
  len_ft <- length(frequency_table_as_list)
  weights <- vector(mode="double", length=len_ft) #will hold number of times each different reality was sampled in particle filter
  for(j in 1:len_ft){
    frequency_table_as_list[[j]] <- as.list(strsplit(frequency_table_as_list[[j]], "], [", fixed=TRUE)[[1]])
    #getting the number of times this reality was sampled in particple filter
    N <- nchar(frequency_table_as_list[[j]][[n_percepts]])
    weights[j] <- substr(frequency_table_as_list[[j]][[n_percepts]], N-3, N-1)
  } #now frequency_table_as_list is a list of lists
  matches <- regmatches(weights, gregexpr("[[:digit:]]+", weights))
  weights <- as.numeric(unlist(matches))
  
  return(list("frequency_table_as_list" = frequency_table_as_list, "weights" = weights))
}

visualize_reality <- function(data){
  data$gt_R <- clean(data$gt_R) #gt_R is a list. it has one element, gt_R[[1]] is characters.
  data$gt_R[[1]] <- substr(data$gt_R[[1]], start=2, stop=nchar(data$gt_R[[1]])-1) #removed extra brackets
  #gsub(pattern = "\\", replacement="",data$gt_R[[1]], fixed = TRUE) #not working
  #because if reality is empty, have to split based on ", S"
  #temp <- as.list(strsplit(data$gt_R[[1]], "], [", fixed=TRUE)[[1]])
  gt_R <- as.list(strsplit(data$gt_R[[1]], "], ", fixed=TRUE)[[1]])
  
  n_percepts <- length(gt_R)
  
  #retrospective metagen
  returned <- dealing_with_frequency_tables(data$frequency.table.PF, n_percepts)
  frequency_table_as_list <- returned$frequency_table_as_list
  weights <- returned$weights
  index <- which(weights==max(weights))
  
  #lesioned metagen
  returned_lesioned <- dealing_with_frequency_tables(data$frequency.table.lesioned.PF, n_percepts)
  frequency_table_as_list_lesioned <- returned_lesioned$frequency_table_as_list
  weights_lesioned <- returned_lesioned$weights
  index_lesioned <- which(weights_lesioned==max(weights_lesioned))
  
  # #double check that the weights sum to number of particles
  # num_particles <- 100
  # #assertthat(sum(weights) == num_particles)
  
  #n_frames <- 10
  category_names = c("person","bicycle","car","motorcycle","airplane")
  num_categories = length(category_names)
  
  gt_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
  perceived_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
  frequency_table_2d_retrospective <- matrix(0, nrow = num_categories, ncol = n_percepts)
  frequency_table_2d_lesioned <- matrix(0, nrow = num_categories, ncol = n_percepts)
  
  matches <- regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data)))
  percepts_list <- unlist(regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data))))
  #percepts_list <- grep(text = gregexpr("percept[[:digit:]]+"), colnames(data), value=TRUE)
  #count up how many objects of each category in each percept and tally it in matrix
  for(p in 1:n_percepts){
    for(cat in 1:num_categories){
      gt_reality[cat,p] <- str_count(gt_R[[p]], pattern = category_names[cat])
      perceived_reality[cat,p] <- str_count(data[[percepts_list[p]]], pattern = category_names[cat])
      frequency_table_2d_retrospective[cat,p] <- str_count(frequency_table_as_list[index][[1]][[p]], pattern = category_names[cat])
      frequency_table_2d_lesioned[cat,p] <- str_count(frequency_table_as_list_lesioned[index_lesioned][[1]][[p]], pattern = category_names[cat])
    }
    #for perceived_reality (used in thresholding), need to know how many frames this percept has
    #Word "Any" is before every frame. if 10 frames, any shows up 10 times
    n_frames <- str_count(data[[percepts_list[p]]], pattern = "Any")
    perceived_reality[,p] <- perceived_reality[,p]/n_frames
  }
  #naive realist says if I saw it in any frame, its there
  naive_reality <- 1*(perceived_reality>0)
  
  # gt_reality
  # perceived_reality
  retrospective_metagen <- frequency_table_2d_retrospective
  lesioned_metagen <- frequency_table_2d_lesioned
  
  cutoff <- 0.5
  thresholded <- perceived_reality
  thresholded[thresholded < cutoff] <- 0
  thresholded[thresholded >= cutoff] <- 1
  
  ################################################
  #all this for online metagen
  frequency_table_2d_online <-
    matrix(0, nrow = num_categories, ncol = n_percepts)
  
  list <-
    grep('frequency.table.PF.after.p', colnames(data), value = TRUE)
  for (n_perc in 1:n_percepts) {
    #all this stuff is for online metagen
    ft <- data[[list[n_perc]]]
    returned <- dealing_with_frequency_tables(ft, n_perc)
    frequency_table_as_list <- returned$frequency_table_as_list
    weights <- returned$weights
    index_online = which(weights == max(weights)) #going by mode over this reality and previous ones
    #might be better to go with mode over just this reality rather than previous ones
    
    vec <- rep(0, num_categories)
    for (cat in 1:num_categories) {
      vec[cat] <-
        str_count(frequency_table_as_list[index_online][[1]][[n_perc]], pattern = category_names[cat])
    }
    frequency_table_2d_online[,n_perc] <- vec
  }
  
  ################################################
  
  #flatten matrices
  gt_reality <- as.vector(t(gt_reality))
  perceived_reality <- as.vector(t(perceived_reality))
  thresholded <- as.vector(t(thresholded))
  frequency_table_retrospective <- as.vector(t(frequency_table_2d_retrospective))
  frequency_table_online <- as.vector(t(frequency_table_2d_online))
  frequency_table_lesioned <- as.vector(t(frequency_table_2d_lesioned))
  
  x <- seq(1,n_percepts)
  y <- category_names
  df <- expand.grid(X=x, Y=y)
  df$gt_reality <- gt_reality
  df$perceived_reality <- perceived_reality
  df$thresholded <- thresholded
  df$frequency_table_retrospective <- frequency_table_retrospective
  df$frequency_table_online <- frequency_table_2d_online
  df$frequency_table_lesioned <- frequency_table_2d_lesioned
  
  # Heatmap 
  #low="black", high="white", in scale_fill_gradient
  p1 <- ggplot(df, aes(X, Y, fill= gt_reality)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Ground-truth realities") +
    xlab("Realities") + ylab("Categories")
  p2 <- ggplot(df, aes(X, Y, fill= perceived_reality)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Percepts") +
    xlab("Percepts presented") + ylab("Categories")
  p3 <- ggplot(df, aes(X, Y, fill= frequency_table_retrospective)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Inferred realities retrospective MetaGen") +
    xlab("Percepts presented") + ylab("Categories")
  p4 <- ggplot(df, aes(X, Y, fill= frequency_table_online)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Inferred realities online MetaGen") +
    xlab("Percepts presented") + ylab("Categories")
  p5 <- ggplot(df, aes(X, Y, fill= frequency_table_lesioned)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Inferred realities lesioned MetaGen") +
    xlab("Percepts presented") + ylab("Categories")
  p6 <- ggplot(df, aes(X, Y, fill= thresholded)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Realities from thresholding percepts") + 
    xlab("Percepts presented") + ylab("Categories")
  
  par(mfrow=c(1,6))
  
  all <- align_plots(p1, p2, p3, p4, p5, p6, align="hv", axis="tblr")
  ggdraw(all[[1]])
  ggdraw(all[[2]])
  ggdraw(all[[3]])
  ggdraw(all[[4]])
  ggdraw(all[[5]])
  ggdraw(all[[6]])
}



##########################################################################
##Now I want to make partial plots for every 10 percepts or so

# list <- grep('frequency.table.PF.after.p', colnames(data), value=TRUE)
# n <- 10 #number of new percepts per each graph
# sequence <- seq(n, n_percepts, n)
# for (i in 1:length(sequence)){
#   i=5
#   n_perc <- sequence[i] #shortened number of percepts
#   ft <- data[[list[n_perc]]]
#   returned <- dealing_with_frequency_tables(ft, n_perc)
#   frequency_table_as_list <- returned$frequency_table_as_list
#   weights <- returned$weights
#   len_ft <- length(frequency_table_as_list)
#   
#   frequency_table_2d <- matrix(0, nrow = num_categories, ncol = n_perc)
#   #making 3d matrix for frequency table of realities
#   some_zeros <- rep(0, len_ft*num_categories*n_percepts)
#   frequency_table_3d <- array(some_zeros, c(num_categories, n_percepts, len_ft))
#   
#   #count up how many objects of each category in each percept and tally it in matrix
#   for(p in 1:n_perc) {
#     for (cat in 1:num_categories) {
#       for (j in 1:len_ft) {
#         frequency_table_3d[cat, p, j] <-
#           weights[j] * str_count(frequency_table_as_list[j][[1]][[p]], pattern = category_names[cat])
#       }
#       frequency_table_2d[cat, p] <- sum(frequency_table_3d[cat, p, ])
#     }
#   }
#   frequency_table_2d <- frequency_table_2d / sum(weights) #sum(weights) should be num_particles
#   
#   #pad with 2s to get to n_percepts columns
#   mat_2 <- matrix(2, nrow = num_categories, ncol = n_percepts - n_perc)
#   frequency_table_2d <- cbind(frequency_table_2d, mat_2)
#   
#   #flatten matrix
#   frequency_table <- as.vector(t(frequency_table_2d))
#   
#   x <- seq(1,n_percepts) #total number percepts
#   y <- category_names
#   df <- expand.grid(X=x, Y=y)
#   df$frequency_table <- frequency_table
#   
#   p <- ggplot(df, aes(X, Y, fill= frequency_table)) + 
#     geom_tile() +
#     scale_fill_gradient(low="black", high="white", limits=c(0, 2)) +
#     ggtitle("Partial inferred realities")
#   
#   ggdraw(p)
# }



