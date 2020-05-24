#This is for parsing inferred reality of the ambiguous percept.
#Takes a data

#use this to get the dealing_with_frequency_tables function
source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy.R")

parse_ambiguous_percept <- function(data){
  
  category_names = c("person","bicycle","car","motorcycle","airplane")
  num_categories <- length(category_names)
  n_percepts <- 51
  data$ambiguous_percept
  returned <- dealing_with_frequency_tables(data$frequency.table.PF.ambiguous_percept, n_percepts)
  frequency_table_as_list <- returned$frequency_table_as_list
  weights <- returned$weights
  index <- which(weights==max(weights)) #doing this based on mode over this and previous
  
  last_reality <- rep(0, length = num_categories)
  for (cat in 1:num_categories) {
    last_reality[cat] <-
      last_reality[cat] +
      str_count(frequency_table_as_list[index][[1]][[n_percepts]], pattern = category_names[cat])
  }
  
  # #might be better to go with mode over just this reality than over everything previous as well
  # len_ft <- length(frequency_table_as_list)
  # frequency_table_last_percept <- rep(0, length = num_categories)
  # for (cat in 1:num_categories) {
  #   for (i in 1:len_ft) {
  #     frequency_table_last_percept[cat] <-
  #       frequency_table_last_percept[cat] +
  #       weights[i] * str_count(frequency_table_as_list[i][[1]][[n_percepts]], pattern = category_names[cat])
  #   }
  # }
  # frequency_table_last_percept <- frequency_table_last_percept/sum(weights)
  
  to_return <- category_names[last_reality==1]
  to_return <- paste(to_return, collapse = ' ')
  #now have to repeat it 50 times to make a dataframe
  return(to_return)
}
