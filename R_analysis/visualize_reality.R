library(dplyr)
library(stringr)
library(ggplot2)
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

visualize_reality <- function(data){
  data$gt_R <- clean(data$gt_R) #gt_R is a list. it has one element, gt_R[[1]] is characters.
  data$gt_R[[1]] <- substr(data$gt_R[[1]], start=2, stop=nchar(data$gt_R[[1]])-1) #removed extra brackets
  #gsub(pattern = "\\", replacement="",data$gt_R[[1]], fixed = TRUE) #not working
  #because if reality is empty, have to split based on ", S"
  #temp <- as.list(strsplit(data$gt_R[[1]], "], [", fixed=TRUE)[[1]])
  gt_R <- as.list(strsplit(data$gt_R[[1]], "], ", fixed=TRUE)[[1]])
  
  num_percepts <- length(gt_R)
  category_names = c("person","bicycle","car","motorcycle","airplane")
  num_categories = length(category_names)
  
  gt_reality <- matrix(0, nrow = num_categories, ncol = num_percepts)
  perceived_reality <- matrix(0, nrow = num_categories, ncol = num_percepts)
  
  matches <- regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data)))
  percepts_list <- unlist(regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data))))
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
  thresholded <- 1*(perceived_reality>=0.5)
  
  ################################################
  #all this for online metagen
  frequency_table_2d_online <-
    matrix(0, nrow = num_categories, ncol = num_percepts)
  frequency_table_2d_retrospective <-
    matrix(0, nrow = num_categories, ncol = num_percepts)
  frequency_table_2d_lesioned <-
    matrix(0, nrow = num_categories, ncol = num_percepts)
  
  list_online <- grep('online.mode.realities.PF.after.p', colnames(data), value=TRUE)
  list_retrospective <- grep('retrospective.mode.realities.PF.after.p', colnames(data), value=TRUE)
  list_lesioned <- grep('lesioned.mode.realities.PF.after.p', colnames(data), value=TRUE)
  
  for (n_perc in 1:num_percepts) {
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
    frequency_table_2d_online[,n_perc] <- online
    frequency_table_2d_retrospective[,n_perc] <- retrospective
    frequency_table_2d_lesioned[,n_perc] <- lesioned
  }
  
  ################################################
  
  #flatten matrices
  gt_reality <- as.vector(t(gt_reality))
  perceived_reality <- as.vector(t(perceived_reality))
  thresholded <- as.vector(t(thresholded))
  frequency_table_retrospective <- as.vector(t(frequency_table_2d_retrospective))
  frequency_table_online <- as.vector(t(frequency_table_2d_online))
  frequency_table_lesioned <- as.vector(t(frequency_table_2d_lesioned))
  
  x <- seq(1,num_percepts)
  y <- category_names
  df <- expand.grid(X=x, Y=y)
  df$gt_reality <- gt_reality
  df$perceived_reality <- perceived_reality
  df$thresholded <- thresholded
  df$frequency_table_retrospective <- frequency_table_retrospective
  df$frequency_table_online <- frequency_table_online
  df$frequency_table_lesioned <- frequency_table_lesioned
  
  mutated <- df %>% mutate(MetaGenWasRight=(gt_reality==frequency_table_retrospective),ThreshWasRight=(gt_reality==thresholded))
  
  mutated %>% ggplot(aes(X,Y,fill=MetaGenWasRight))+geom_tile()+theme(aspect.ratio=1)
  
  mutated %>% ggplot(aes(X,Y,fill=ThreshWasRight))+geom_tile()+theme(aspect.ratio=1) 
  
  coded_mistakes <- function(gt_reality, frequency_table_retrospective){
    to_return <- rep(0, length(gt_reality))
    for (i in 1:length(gt_reality)){
      if(gt_reality[i]==frequency_table_retrospective[i]){
        to_return[i]=0
      } else if(gt_reality[i]==1 & frequency_table_retrospective[i]==0){
        to_return[i]=1# miss
      } else{
        to_return[i]=2 #FA
      }
    }
    return(to_return)
  }
  #0 if they're the same, 1 if it's a miss, 2 if it's a false alarm
  mutated <- df %>% mutate(MetaGenCodedMistakes=coded_mistakes(gt_reality, frequency_table_retrospective), 
                           ThreshWasCodedMistakes=coded_mistakes(gt_reality, thresholded))
  
  mutated %>% ggplot(aes(X,Y,fill=factor(MetaGenCodedMistakes)))+geom_tile()+theme(aspect.ratio=1)
  
  mutated %>% ggplot(aes(X,Y,fill=factor(ThreshWasCodedMistakes)))+geom_tile()+theme(aspect.ratio=1) 
  
  # Heatmap 
  #low="black", high="white", in scale_fill_gradient
  p1 <- ggplot(df, aes(X, Y, fill= gt_reality)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Ground-truth realities") +
    xlab("Realities") + ylab("Categories") + theme(aspect.ratio=1)
  p2 <- ggplot(df, aes(X, Y, fill= perceived_reality)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Percepts") +
    xlab("Percepts presented") + ylab("Categories") + theme(aspect.ratio=1)
  p3 <- ggplot(df, aes(X, Y, fill= frequency_table_retrospective)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Inferred realities retrospective MetaGen") +
    xlab("Percepts presented") + ylab("Categories") + theme(aspect.ratio=1)
  # p4 <- ggplot(df, aes(X, Y, fill= frequency_table_online)) + 
  #   geom_tile() +
  #   scale_fill_gradient(limits=c(0, 1)) +
  #   ggtitle("Inferred realities online MetaGen") +
  #   xlab("Percepts presented") + ylab("Categories")
  # p5 <- ggplot(df, aes(X, Y, fill= frequency_table_lesioned)) + 
  #   geom_tile() +
  #   scale_fill_gradient(limits=c(0, 1)) +
  #   ggtitle("Inferred realities lesioned MetaGen") +
  #   xlab("Percepts presented") + ylab("Categories")
  p6 <- ggplot(df, aes(X, Y, fill= thresholded)) + 
    geom_tile() +
    scale_fill_gradient(limits=c(0, 1)) +
    ggtitle("Realities from thresholding percepts") + 
    xlab("Percepts presented") + ylab("Categories") + theme(aspect.ratio=1)
  
  par(mfrow=c(1,6))
  
  all <- align_plots(p1, p2, p3, p4, p5, p6, align="hv", axis="tblr")
  ggdraw(all[[1]])
  ggdraw(all[[2]])
  ggdraw(all[[3]])
  #ggdraw(all[[4]])
  #ggdraw(all[[5]])
  ggdraw(all[[6]])
}

##########################################################################
