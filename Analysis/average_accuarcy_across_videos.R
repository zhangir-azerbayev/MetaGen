library(tidyverse)
library(boot)
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_18/full_versions/")

online_0 = read_csv("shuffle_0/similarity2D.csv")
online_1 = read_csv("shuffle_1/similarity2D.csv")
online_2 = read_csv("shuffle_2/similarity2D.csv")
online_3 = read_csv("shuffle_3/similarity2D.csv")

merged1 = merge(online_0, online_1, by = c("video", "sim_NN_fitted", "sim_NN_input"), suffixes = c(".0",".1"))
merged2 = merge(online_2, online_3, by = c("video", "sim_NN_fitted", "sim_NN_input"), suffixes = c(".2",".3"))
merged = merge(merged1, merged2, by = c("video", "sim_NN_fitted", "sim_NN_input"))

#####################################################################################
#helper functions for bootstrapping
compute_mean <- function(DataList, indices){
  sampled_data = DataList[indices]
  return(mean(sampled_data))
}

get_ci <- function(data_column){
  #if there's an NaN, bootstrapping won't work 
  if (is.nan(data_column[1])){
    return(c(NaN, NaN))
  }
  simulations <- boot(data = data_column, statistic=compute_mean, R=1000)
  results <- boot.ci(simulations) #type doesn't seem to work
  lower <- results$percent[4]
  upper <- results$percent[5]
  return(c(lower, upper))
}

#####################################################################################
#just eyeball the variance
merged %>% group_by(video < 51) %>% summarize(m.0 = mean(sim_online.0), m.1 = mean(sim_online.1), m.2 = mean(sim_online.2), m.3 = mean(sim_online.3))
merged %>% group_by(video < 51) %>% summarize(m.0 = mean(sim_retrospective.0), m.1 = mean(sim_retrospective.1), m.2 = mean(sim_retrospective.2), m.3 = mean(sim_retrospective.3))


df <- merged %>% mutate(sim_online = (sim_online.0 + sim_online.1 + sim_online.2 + sim_online.3)/4,
                        sim_retro = (sim_retrospective.0 + sim_retrospective.1 + sim_retrospective.2 + sim_retrospective.3)/4)

df <- df %>% select(video, sim_online, sim_retro, sim_NN_fitted, sim_NN_input) %>%
  rename(sim_fitted = sim_NN_fitted, sim_input = sim_NN_input)

df <- df %>% mutate(grouping = video > 50)

temp <- df %>% group_by(grouping) %>%
  summarize(lower_online = get_ci(sim_online)[1],
         lower_retro = get_ci(sim_retro)[1],
         lower_fitted = get_ci(sim_fitted)[1],
         lower_input = get_ci(sim_input)[1],
         upper_online = get_ci(sim_online)[2],
         upper_retro = get_ci(sim_retro)[2],
         upper_fitted = get_ci(sim_fitted)[2],
         upper_input = get_ci(sim_input)[2],
         mean_online = mean(sim_online),
         mean_retro = mean(sim_retro),
         mean_fitted = mean(sim_fitted),
         mean_input = mean(sim_input))# %>%

first_half <- temp %>% filter(grouping==FALSE) %>% select(-grouping)
second_half <- temp %>% filter(grouping) %>% select(-grouping)

df1 <- first_half %>%
  pivot_longer(everything(),
               names_to = c(".value", "model"),
               names_pattern = "(.+)_(.+)"
  )
  
df2 <- second_half %>%
  pivot_longer(everything(),
               names_to = c(".value", "model"),
               names_pattern = "(.+)_(.+)"
  )


df1 <- df1 %>% mutate(group = "A")
df2 <- df2 %>% mutate(group = "B")
to_plot <- rbind(df1, df2)

pd <- position_dodge(0.1)
ggplot(
  to_plot, 
  aes(x = group, y = mean, color = model)
  ) + 
  geom_errorbar(width = 0.1, aes(ymin = lower, ymax = upper), position = pd) +
  geom_point(position = pd) +
  ylim(0,1)




