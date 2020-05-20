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

# output111 <- read_delim("outputOld_model_printing_fts.csv",
#                         "&", escape_double = FALSE, trim_ws = TRUE)
output111 <- read_delim("output111.csv",
                        "&", escape_double = FALSE, trim_ws = TRUE)
data <- output111
names(data)<-make.names(names(data),unique = TRUE)

data$mode.realities.PF <- data$mode.realities.PF  %>%
  lapply(function(x){gsub(pattern = "Array{String,N} where N", replacement="",x, fixed = TRUE)})

data$gt_R <- clean(data$gt_R) #gt_R is a list. it has one element, gt_R[[1]] is characters.
data$gt_R[[1]] <- substr(data$gt_R[[1]], start=2, stop=nchar(data$gt_R[[1]])-1) #removed extra brackets
#gsub(pattern = "\\", replacement="",data$gt_R[[1]], fixed = TRUE) #not working
#because if reality is empty, have to split based on ", S"
#temp <- as.list(strsplit(data$gt_R[[1]], "], [", fixed=TRUE)[[1]])
gt_R <- as.list(strsplit(data$gt_R[[1]], "], ", fixed=TRUE)[[1]])

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

n_percepts <- length(gt_R)

#retrospective metagen
returned <- dealing_with_frequency_tables(data$frequency.table.PF, n_percepts)
frequency_table_as_list <- returned$frequency_table_as_list
weights <- returned$weights
len_ft <- length(frequency_table_as_list)

#lesioned metagen
returned_lesioned <- dealing_with_frequency_tables(data$frequency.table.lesioned.PF, n_percepts)
frequency_table_as_list_lesioned <- returned_lesioned$frequency_table_as_list
weights_lesioned <- returned_lesioned$weights
len_ft_lesioned <- length(frequency_table_as_list_lesioned)

# #double check that the weights sum to number of particles
# num_particles <- 100
# #assertthat(sum(weights) == num_particles)


n_frames <- 10
category_names = c("person","bicycle","car","motorcycle","airplane")
num_categories = length(category_names)


gt_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
perceived_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
frequency_table_2d <- matrix(0, nrow = num_categories, ncol = n_percepts)
frequency_table_2d_lesioned <- matrix(0, nrow = num_categories, ncol = n_percepts)
#making 3d matrix for frequency table of realities
some_zeros <- rep(0, len_ft*num_categories*n_percepts)
frequency_table_3d <- array(some_zeros, c(num_categories, n_percepts, len_ft))
some_zeros <- rep(0, len_ft_lesioned*num_categories*n_percepts)
frequency_table_3d_lesioned <- array(some_zeros, c(num_categories, n_percepts, len_ft_lesioned))

matches <- regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data)))
percepts_list <- unlist(regmatches(colnames(data), gregexpr("percept[[:digit:]]+", colnames(data))))
#percepts_list <- grep(text = gregexpr("percept[[:digit:]]+"), colnames(data), value=TRUE)
#count up how many objects of each category in each percept and tally it in matrix
for(p in 1:n_percepts){
  for(cat in 1:num_categories){
    gt_reality[cat,p] <- str_count(gt_R[[p]], pattern = category_names[cat])
    perceived_reality[cat,p] <- str_count(data[[percepts_list[p]]], pattern = category_names[cat])
    for(j in 1:len_ft){
      frequency_table_3d[cat,p,j] <- weights[j]*str_count(frequency_table_as_list[j][[1]][[p]], pattern = category_names[cat])
    }
    for(j in 1:len_ft_lesioned){
      frequency_table_3d_lesioned[cat,p,j] <- weights_lesioned[j]*str_count(frequency_table_as_list_lesioned[j][[1]][[p]], pattern = category_names[cat])
    }
    frequency_table_2d[cat,p] <- sum(frequency_table_3d[cat,p,])
    frequency_table_2d_lesioned[cat,p] <- sum(frequency_table_3d_lesioned[cat,p,])
  }
}
perceived_reality <- perceived_reality/n_frames
frequency_table_2d <- frequency_table_2d/sum(weights) #sum(weights) should be num_particles
frequency_table_2d_lesioned <- frequency_table_2d_lesioned/sum(weights_lesioned) #sum(weights) should be num_particles


gt_reality
perceived_reality
retrospective_metagen <- frequency_table_2d
lesioned_metagen <- frequency_table_2d_lesioned


num_percepts = length(gt_R)
E_online_metagen <- rep(0, num_percepts)
E_retrospective_metagen <- rep(0, num_percepts)
E_lesioned_metagen <- rep(0, num_percepts)
E_perceived_reality <- rep(0, num_percepts)
E_threshold <- rep(0, num_percepts)
list <- grep('frequency.table.PF.after.p', colnames(data), value=TRUE)

for(n_perc in 1:num_percepts){
  
  #all this stuff is for online metagen
  ft <- data[[list[n_perc]]]
  returned <- dealing_with_frequency_tables(ft, n_perc)
  frequency_table_as_list <- returned$frequency_table_as_list
  weights <- returned$weights
  len_ft <- length(frequency_table_as_list)
  
  mat <- matrix(0, nrow = num_categories, ncol = len_ft)
  for (cat in 1:num_categories) {
    for (j in 1:len_ft) {
      mat[cat, j] <-
        weights[j]*str_count(frequency_table_as_list[j][[1]][[n_perc]], pattern = category_names[cat])
    }
  }
  online_metagen <- rowSums(mat)/sum(weights)
  

  E_online_metagen[n_perc] <- sum(abs(gt_reality[,n_perc] - online_metagen))
  E_retrospective_metagen[n_perc] <- sum(abs(gt_reality[,n_perc] - retrospective_metagen[,n_perc]))
  E_lesioned_metagen[n_perc] <- sum(abs(gt_reality[,n_perc] - lesioned_metagen[,n_perc]))
  E_perceived_reality[n_perc] <- sum(abs(gt_reality[,n_perc] - perceived_reality[,n_perc]))
  E_threshold[n_perc] <- sum(abs(gt_reality[,n_perc] - perceived_reality[,n_perc]>0.5))
  
}

E_online_metagen
E_retrospective_metagen
E_lesioned_metagen
E_perceived_reality #could get noisiest percepts by finding location of max value in E_perceived_reality
E_threshold

#turn errors into accuracy
A_online_metagen = (num_categories-E_online_metagen)/num_categories
A_retrospective_metagen = (num_categories-E_retrospective_metagen)/num_categories
A_lesioned_metagen = (num_categories-E_lesioned_metagen)/num_categories
A_perceived_reality = (num_categories-E_perceived_reality)/num_categories
A_threshold = (num_categories-E_threshold)/num_categories
x <- 1:num_percepts
toPlot <- data.frame(x, A_retrospective_metagen, A_lesioned_metagen, A_online_metagen, A_perceived_reality, A_threshold)

# library("tidyverse")
# df <- toPlot %>%
#   select(x, A_online_metagen, A_retrospective_metagen, A_perceived_reality, A_threshold) %>%
#   gather(key = "variable", value = "value", -x)
# ggplot(df, aes(x = x, y = value)) + 
#   geom_line(aes(color = variable)) + 
#   scale_color_manual(values = c("darkred", "steelblue", "green", "orange")) +
#   xlab("Number of Percepts Observed") + ylab("Accuracy")

ggplot(toPlot, aes(x=x)) +
  geom_line(aes(y = A_online_metagen), color = "darkred") +
  #geom_line(aes(y = A_retrospective_metagen), color="steelblue") +
  #geom_line(aes(y = A_lesioned_metagen), color="blue") +
  #geom_line(aes(y = A_perceived_reality), color="green") +
  #geom_line(aes(y = A_threshold), color="orange") +
  xlab("Number of Percepts Observed") + ylab("Accuracy")