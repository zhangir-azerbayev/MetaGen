#This function will return a dataframe with accuracy scores for several models.

library(dplyr)
library(readr)
library(stringr)

parse_last_percept <- function(ft) {
  if (str_detect(ft, fixed("], ["))) {
    df <- as.data.frame(str_locate_all(ft, fixed("], [")))
    start = df$end[nrow(df)]
    last_percept = substring(ft, start, nchar(ft))
  } else{
    last_percept = ft
  }
  return(last_percept)
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
  
  num_percepts <- length(gt_R)
  
  #n_frames <- 10
  category_names = c("person","bicycle","car","motorcycle","airplane")
  num_categories = length(category_names)
  
  gt_reality <- matrix(0, nrow = num_categories, ncol = num_percepts)
  perceived_reality <- matrix(0, nrow = num_categories, ncol = num_percepts)
  
  matches <- regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data)))
  percepts_list <- unlist(matches)
  #percepts_list <- grep(text = gregexpr("percept[[:digit:]]+"), colnames(data), value=TRUE)
  #count up how many objects of each category in each percept and tally it in matrix
  for(p in 1:num_percepts){
    for(cat in 1:num_categories){
      gt_reality[cat,p] <- str_count(gt_R[[p]], pattern = category_names[cat])
      perceived_reality[cat,p] <- str_count(data[[percepts_list[p]]], pattern = category_names[cat])
    }
    #for perceived_reality (used in thresholding), need to know how many frames this percept has
    #Word "Any" is before every frame. if 10 frames, any shows up 10 times
    n_frames <- str_count(data[[percepts_list[p]]], pattern = "Any")
    perceived_reality[,p] <- perceived_reality[,p]/n_frames
  }
  #naive realist says if I saw it in any frame, its there
  naive_reality <- 1*(perceived_reality>0)
  
  num_percepts = num_percepts
  A_online_metagen <- rep(0, num_percepts)
  A_retrospective_metagen <- rep(0, num_percepts)
  A_lesioned_metagen <- rep(0, num_percepts)
  A_naive_reality <- rep(0, num_percepts)
  A_threshold <- rep(0, num_percepts)
  
  #how bad was the perceived noise?
  perceived_noise <- rep(0, num_percepts)
  
  list_online <- grep('online.mode.realities.PF.after.p', colnames(data), value=TRUE)
  list_retrospective <- grep('retrospective.mode.realities.PF.after.p', colnames(data), value=TRUE)
  list_lesioned <- grep('lesioned.mode.realities.PF.after.p', colnames(data), value=TRUE)
  
  for(n_perc in 1:num_percepts){

    online_most_recent_percept <- parse_last_percept(data[[list_online[n_perc+1]]])
    retrospective_most_recent_percept <- parse_last_percept(data[[list_retrospective[n_perc+1]]])
    lesioned_most_recent_percept <- parse_last_percept(data[[list_lesioned[n_perc+1]]])
    
    online <- rep(0, num_categories)
    retrospective <- rep(0, num_categories)
    lesioned <- rep(0, num_categories)
    for (cat in 1:num_categories) {
      online[cat] <- str_count(online_most_recent_percept, pattern = category_names[cat])
      retrospective[cat] <- str_count(retrospective_most_recent_percept, pattern = category_names[cat])
      lesioned[cat] <- str_count(lesioned_most_recent_percept, pattern = category_names[cat])
    }
    
    #how unusual (or bad) was each percept? How different was it from reality?
    perceived_noise[n_perc] <- sum(abs(gt_reality[,n_perc] - perceived_reality[,n_perc]))/num_categories
    
    #accuracy is does the gt_reality equal inferred one or not
    A_online_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - online))==0)
    A_retrospective_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - retrospective))==0)
    A_lesioned_metagen[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - lesioned))==0)
    A_naive_reality[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - naive_reality[,n_perc]))==0)
    A_threshold[n_perc] <- 1*(sum(abs(gt_reality[,n_perc] - (perceived_reality[,n_perc]>=0.5)))==0)
  }
  
  percept_number <- 1:num_percepts
  diff_between_retro_and_threshold <- (sum(A_retrospective_metagen)-sum(A_threshold))/num_percepts
  toPlot <- data.frame(percept_number, A_retrospective_metagen, A_lesioned_metagen, A_online_metagen, A_naive_reality, A_threshold, perceived_noise, diff_between_retro_and_threshold)
  
  #simply add NAs to beginning of each vector to make up for percept 0
  padding <- c(0, rep(NA, ncol(toPlot)-1))
  toPlot <- rbind(padding, toPlot)
  
  return(toPlot)
}
