library(tidyverse)
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/")
data = read_delim("similarity.csv", delim='&')

mean(data$sim_lesioned)
mean(data$sim_online)
mean(data$sim_retrospective)

mean(data$sim_lesioned[51:100])
mean(data$sim_online[51:100])
mean(data$sim_retrospective[51:100])

df = gather(data, "model", "value", c(sim_online, sim_lesioned, sim_retrospective))
df = gather(data, "model", "value", c(sim_online))


jitter <- position_jitter(width = 5, height = )
p <- ggplot(
  df,
  aes(
    x = video,
    y = value,
    color = model
  )
) + #geom_point(position = jitter, alpha = 0.5) +
  geom_point() +
  geom_smooth(method='lm', formula= y~x) + theme(aspect.ratio=1)
p
