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

output111 <- read_delim("outputOld_model_printing_fts.csv",
                        "&", escape_double = FALSE, trim_ws = TRUE)
# output111 <- read_delim("output111.csv",
#                         "&", escape_double = FALSE, trim_ws = TRUE)
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
returned <- dealing_with_frequency_tables(data$frequency.table.PF, n_percepts)
frequency_table_as_list <- returned$frequency_table_as_list
weights <- returned$weights
len_ft <- length(frequency_table_as_list)

# #double check that the weights sum to number of particles
# num_particles <- 100
# #assertthat(sum(weights) == num_particles)




n_frames <- 10
n_percepts <- length(gt_R)
category_names = c("person","bicycle","car","motorcycle","airplane")
num_categories = length(category_names)

gt_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
perceived_reality <- matrix(0, nrow = num_categories, ncol = n_percepts)
frequency_table_2d <- matrix(0, nrow = num_categories, ncol = n_percepts)
#making 3d matrix for frequency table of realities
some_zeros <- rep(0, len_ft*num_categories*n_percepts)
frequency_table_3d <- array(some_zeros, c(num_categories, n_percepts, len_ft))

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
    frequency_table_2d[cat,p] <- sum(frequency_table_3d[cat,p,])
  }
}
perceived_reality <- perceived_reality/n_frames
frequency_table_2d <- frequency_table_2d/sum(weights) #sum(weights) should be num_particles

gt_reality
perceived_reality
frequency_table_2d

make_E <- function(n_columns_to_compare, gt_reality, to_compare){
  Es <- rep(0, n_columns_to_compare)
  for(i in 1:n_columns_to_compare){
    Es[i] <- sum(abs(gt_reality[,i] - to_compare[,i])) #not doing MSE because lots are 0.2 and we don't want to make that smaller
  }
  E <- sum(Es)/n_columns_to_compare #Just because we don't want MSE going up after every percept. want average across percepts
  return(E)
}


num_percepts = length(gt_R)
ME_ft <- rep(0, num_percepts)
ME_perceived_reality <- rep(0, num_percepts)
list <- grep('frequency.table.PF.after.p', colnames(data), value=TRUE)

for(n_perc in 1:num_percepts){
  
  ft <- data[[list[n_perc]]]
  returned <- dealing_with_frequency_tables(ft, n_perc)
  frequency_table_as_list <- returned$frequency_table_as_list
  weights <- returned$weights
  len_ft <- length(frequency_table_as_list)
  
  frequency_table_2d <- matrix(0, nrow = num_categories, ncol = n_perc)
  #making 3d matrix for frequency table of realities
  some_zeros <- rep(0, len_ft*num_categories*n_percepts)
  frequency_table_3d <- array(some_zeros, c(num_categories, n_percepts, len_ft))
  
  #count up how many objects of each category in each percept and tally it in matrix
  for(p in 1:n_perc) {
    for (cat in 1:num_categories) {
      for (j in 1:len_ft) {
        frequency_table_3d[cat, p, j] <-
          weights[j] * str_count(frequency_table_as_list[j][[1]][[p]], pattern = category_names[cat])
      }
      frequency_table_2d[cat, p] <- sum(frequency_table_3d[cat, p, ])
    }
  }
  frequency_table_2d <- frequency_table_2d / sum(weights) #sum(weights) should be num_particles
  
  ME_ft[n_perc] <- make_E(n_perc, gt_reality, frequency_table_2d)
  ME_perceived_reality[n_perc] <- make_E(n_perc, gt_reality, as.matrix(perceived_reality[,1:n_perc]))
}

ME_ft
ME_perceived_reality
