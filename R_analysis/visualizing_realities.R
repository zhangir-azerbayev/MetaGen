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

output111 <- read_delim("output111.csv", 
                        "&", escape_double = FALSE, trim_ws = TRUE)
data <- output111
names(data)<-make.names(names(data),unique = TRUE)

data$mode.realities.PF <- data$mode.realities.PF  %>%
  lapply(function(x){gsub(pattern = "Array{String,N} where N", replacement="",x, fixed = TRUE)})
data$frequency.table.PF <- data$frequency.table.PF  %>%
  lapply(function(x){gsub(pattern = "Dict(\"Array{String,N} where N", replacement="",x, fixed = TRUE)})

data$gt_R <- clean(data$gt_R) #gt_R is a list. it has one element, gt_R[[1]] is characters.
data$gt_R[[1]] <- substr(data$gt_R[[1]], start=2, stop=nchar(data$gt_R[[1]])-1) #removed extra brackets
#gsub(pattern = "\\", replacement="",data$gt_R[[1]], fixed = TRUE) #not working
gt_R <- as.list(strsplit(data$gt_R[[1]], "], [", fixed=TRUE)[[1]])

n_percepts <- length(gt_R)

frequency_table_as_list <- as.list(strsplit(data$frequency.table.PF[[1]], "Array{String,N} where N", fixed=TRUE)[[1]])
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

percepts_list <- grep('percept', colnames(data), value=TRUE)
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


#flatten matrix
gt_reality <- as.vector(t(gt_reality)) 
perceived_reality <- as.vector(t(perceived_reality)) 
frequency_table <- as.vector(t(frequency_table_2d))

x <- seq(1,n_percepts)
y <- category_names
df <- expand.grid(X=x, Y=y)
df$gt_reality <- gt_reality
df$perceived_reality <- perceived_reality
df$frequency_table <- frequency_table

# Heatmap 
p1 <- ggplot(df, aes(X, Y, fill= gt_reality)) + 
  geom_tile() +
  ggtitle("Ground-truth realities")
p2 <- ggplot(df, aes(X, Y, fill= perceived_reality)) + 
  geom_tile() +
  ggtitle("Percepts")
p3 <- ggplot(df, aes(X, Y, fill= frequency_table)) + 
  geom_tile() +
  ggtitle("Inferred realities")

all <- align_plots(p1, p2, p3, align="hv", axis="tblr")
ggdraw(all[[1]])
ggdraw(all[[2]])
ggdraw(all[[3]])
