library(dplyr)
library(readr)
library(ggplot2)
library(stringr)

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
data <- output111
names(data)<-make.names(names(data),unique = TRUE)

category_names = c("person","bicycle","car","motorcycle","airplane")

data$gt_R <- clean(data$gt_R) #gt_R is a list. it has one element, gt_R[[1]] is characters.
data$gt_R[[1]] <- substr(data$gt_R[[1]], start=2, stop=nchar(data$gt_R[[1]])-1) #removed extra brackets
gt_R <- as.list(strsplit(data$gt_R[[1]], "], ", fixed=TRUE)[[1]])

#returns a list of each entry in the frequency table and the weights of each
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

#This function takes a single reality or frame of a percept and ouputs a 0 1 vector
#representation of length length(category_names)
turn_words_into_numbers <- function(reality_or_frame, category_names = c("person","bicycle","car","motorcycle","airplane")){
  vector_representation <- rep(0, length(category_names))
  for(cat in 1:length(category_names)){
    vector_representation[cat] <- grepl(category_names[cat], reality_or_frame)
  }
  return(vector_representation)
}

n_percepts <- length(gt_R)
gt_R_numbers <- sapply(gt_R, turn_words_into_numbers)

returned <- dealing_with_frequency_tables(data$frequency.table.PF, n_percepts)
frequency_table_as_list <- returned$frequency_table_as_list
weights <- returned$weights
len_ft <- length(frequency_table_as_list)

sum_over_ft <- matrix(0, nrow = num_categories, ncol = n_percepts)
for(i in 1:len_ft){
  to_add <- sapply(frequency_table_as_list[[i]], turn_words_into_numbers)
  weight <- weights[i]
  sum_over_ft <- sum_over_ft + weight*to_add
}
average_over_ft <- sum_over_ft/sum(weights)

n_columns_to_compare <- 50
MSEs <- rep(0, n_columns_to_compare)
for(i in 1:n_columns_to_compare){
  MSEs[i] <- sum((gt_R_numbers[,i] - average_over_ft[,i])^2)
}
MSE <- sum(MSEs)/n_columns_to_compare
