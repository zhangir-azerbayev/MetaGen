library(dplyr)
library(tidyr)
library(readr)
library(tidyboot)
library(ggplot2)

source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy.R")

# data <-
#   list.files(
#     "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/Simulated_data_and_analysis/May/test",
#     pattern = 'output'
#   ) %>%
#   lapply(function(x) {
#     read.csv(x, header = TRUE, sep = '&')
#   })
##data = data %>% bind_rows

# combined_data <- vector(mode = "list", length = length(data))
# for(i in 1:length(data)){
#   combined_data[[i]] <- cbind(simID = i, accuracy(data[[i]]))
# }
# combined_data <- combined_data %>% bind_rows

data <- read_delim("merged.csv",
                        "&", escape_double = FALSE, trim_ws = TRUE)
names(data)<-str_replace_all(names(data), c(" " = "."))

simID <- 1:dim(data)[1]
data <- cbind(simID, data)

##################################

# data[1,] %>% accuracy()
# apply(data, 1, FUN=accuracy(x))
# 
# accuracy(data[1,])

#Row 3478 is messed up. I think these didn't finish running or something
#and row 3510
# data <- data[-3478,]
data <- na.omit(data)

#I <- nrow(data)
I <- 3477


#Must be a better way to do this
combined_data <- vector(mode = "list", length = I)
for(i in 1:I){
  print(i)
  combined_data[[i]] <- accuracy(data[i,])
}
combined_data <- combined_data %>% bind_rows

df <-
  combined_data %>% gather(
    Model,
    Score,
    A_retrospective_metagen,
    A_lesioned_metagen,
    A_online_metagen,
    A_naive_reality,
    A_threshold
  )

#group_by(percept_number)
MyDataFrame <- df %>% group_by(percept_number, Model) %>% tidyboot_mean(Score)

ggplot(
  MyDataFrame,
  aes(
    x = percept_number,
    y = empirical_stat,
    ymin = ci_lower,
    ymax = ci_upper,
    fill = Model,
    group = Model
  )
) + geom_ribbon() + geom_line()
