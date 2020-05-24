#This function will return a dataframe with accuracy scores for several models.

library(dplyr)
library(readr)
library(stringr)

dealing_with_frequency_tables <- function(ft, n_percepts){
  ft <- ft  %>% lapply(function(x){gsub(pattern = "Dict(\"Array{String,N} where N", replacement="",x, fixed = TRUE)})
  ft <- ft  %>% lapply(function(x){gsub(pattern = "Dict(Array{String,N} where N", replacement="",x, fixed = TRUE)})
  
  frequency_table_as_list <- as.list(strsplit(ft[[1]], "Array{String,N} where N", fixed=TRUE)[[1]])
  len_ft <- length(frequency_table_as_list)
  weights <- vector(mode="double", length=len_ft) #will hold number of times each different reality was sampled in particle filter
  for(j in 1:len_ft){
    frequency_table_as_list[[j]] <- as.list(strsplit(frequency_table_as_list[[j]], "], [", fixed=TRUE)[[1]])
    #getting the number of times this reality was sampled in particple filter
    N <- nchar(frequency_table_as_list[[j]][[n_percepts]])
    #weights[j] <- substr(frequency_table_as_list[[j]][[n_percepts]], N-3, N-1)
    weights[j] <- frequency_table_as_list[[j]][[n_percepts]]
  } #now frequency_table_as_list is a list of lists
  matches <- regmatches(weights, gregexpr("[[:digit:]]+", weights))
  weights <- as.numeric(unlist(matches))
  
  return(list("frequency_table_as_list" = frequency_table_as_list, "weights" = weights))
}

#function for cleaning up Vs
clean <- function(column){
  column <- column %>%
    # lapply(function(x){gsub(pattern = "[", replacement="",x, fixed = TRUE)}) %>%
    # lapply(function(x){gsub(pattern = "]", replacement="",x, fixed = TRUE)}) %>%
    # lapply(function(x){gsub(pattern = ";", replacement="",x, fixed = TRUE)}) %>%
    lapply(function(x){gsub(pattern = "Any", replacement="",x, fixed = TRUE)})
  # lapply(function(x){gsub(pattern = "\\", replacement="",x, fixed = TRUE)})
}


accuracy <- function(data){
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
  frequency_table_2d <- matrix(0, nrow = num_categories, ncol = n_percepts)
  frequency_table_2d_lesioned <- matrix(0, nrow = num_categories, ncol = n_percepts)
  
  matches <- regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data)))
  percepts_list <- unlist(regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data))))
  #percepts_list <- grep(text = gregexpr("percept[[:digit:]]+"), colnames(data), value=TRUE)
  #count up how many objects of each category in each percept and tally it in matrix
  for(p in 1:n_percepts){
    for(cat in 1:num_categories){
      gt_reality[cat,p] <- str_count(gt_R[[p]], pattern = category_names[cat])
      perceived_reality[cat,p] <- str_count(data[[percepts_list[p]]], pattern = category_names[cat])
      frequency_table_2d[cat,p] <- str_count(frequency_table_as_list[index][[1]][[p]], pattern = category_names[cat])
      frequency_table_2d_lesioned[cat,p] <- str_count(frequency_table_as_list_lesioned[index_lesioned][[1]][[p]], pattern = category_names[cat])
    }
    #for perceived_reality (used in thresholding), need to know how many frames this percept has
    #Word "Any" is before every frame. if 10 frames, any shows up 10 times
    n_frames <- str_count(data[[percepts_list[p]]], pattern = "Any")
    print(n_frames)
    perceived_reality[,p] <- perceived_reality[,p]/n_frames
  }
  #naive realist says if I saw it in any frame, its there
  naive_reality <- 1*(perceived_reality>0)
  
  # gt_reality
  # perceived_reality
  retrospective_metagen <- frequency_table_2d
  lesioned_metagen <- frequency_table_2d_lesioned
  
  
  num_percepts = length(gt_R)
  A_online_metagen <- rep(0, num_percepts)
  A_retrospective_metagen <- rep(0, num_percepts)
  A_lesioned_metagen <- rep(0, num_percepts)
  A_naive_reality <- rep(0, num_percepts)
  A_threshold <- rep(0, num_percepts)
  
  #how bad was the perceived noise?
  perceived_noise <- rep(0, num_percepts)
  
  list <- grep('frequency.table.PF.after.p', colnames(data), value=TRUE)
  
  for(n_perc in 1:num_percepts){
    
    #all this stuff is for online metagen
    ft <- data[[list[n_perc]]]
    returned <- dealing_with_frequency_tables(ft, n_perc)
    frequency_table_as_list <- returned$frequency_table_as_list
    weights <- returned$weights
    index_online = which(weights==max(weights)) #going by mode over this reality and previous ones
    #might be better to go with mode over just this reality rather than previous ones
    
    vec <- rep(0, num_categories)
    for (cat in 1:num_categories) {
      vec[cat] <- str_count(frequency_table_as_list[index_online][[1]][[n_perc]], pattern = category_names[cat])
    }
    online_metagen <- vec
    
    #how unusual (or bad) was each percept? How different was it from reality?
    perceived_noise[n_perc] <- sum(abs(gt_reality[,n_perc] - perceived_reality[,n_perc]))
    
    #accuracy is does the gt_reality equal inferred one or not
    A_online_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - online_metagen))==0)
    A_retrospective_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - retrospective_metagen[,n_perc]))==0)
    A_lesioned_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - lesioned_metagen[,n_perc]))==0)
    A_naive_reality[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - naive_reality[,n_perc]))==0)
    A_threshold[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - (perceived_reality[,n_perc]>=0.5)))==0)
  }
  
  percept_number <- 1:num_percepts
  toPlot <- data.frame(percept_number, A_retrospective_metagen, A_lesioned_metagen, A_online_metagen, A_naive_reality, A_threshold, perceived_noise)
  
  return(toPlot)
}
