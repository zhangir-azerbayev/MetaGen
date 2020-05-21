library(dplyr)
library(tidyr)
library(readr)
library(tidyboot)
library(ggplot2)

source("/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/R_analysis/accuracy.R")

data <-
  list.files(
    "/Users/marleneberke/Documents/03_Yale/Projects/001_Mask_RCNN/Simulated_data_and_analysis/May/test",
    pattern = 'output'
  ) %>%
  lapply(function(x) {
    read.csv(x, header = TRUE, sep = '&')
  })
#data = data %>% bind_rows

combined_data <- vector(mode = "list", length = length(data))
for(i in 1:length(data)){
  combined_data[[i]] <- cbind(simID = i, accuracy(data[[i]]))
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
MyDataFrame <- df %>% group_by(x, Model) %>% tidyboot_mean(Score)

ggplot(
  MyDataFrame,
  aes(
    x = x,
    y = empirical_stat,
    ymin = ci_lower,
    ymax = ci_upper,
    fill = Model,
    group = Model
  )
) + geom_ribbon() + geom_line()
