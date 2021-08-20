library(tidyverse)
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/Data")

online = read_delim("online_output.csv", delim='&')
retrospective = read_delim("retrospective_output.csv", delim='&')
lesioned = read_delim("lesioned_output.csv", delim='&')
#similarity = read_delim("similarity.csv", delim='&')
ideal_v_matrix = read_delim("ideal_v_matrix.csv", delim='&')
ground_truth_V = read_delim("ground_truth_V.csv", delim='&')

merged1 = merge(retrospective, lesioned, by = "video_number", suffixes = c(".retro",".lesion"))
merged2 = merge(online, merged1, by = "video_number")
merged3 = merge(merged2, ideal_v_matrix, by = "video_number")
merged4 = merge(merged3, ground_truth_V, by = "video_number")

#to graph every model's version of m_1
df_M = gather(merged4, "M_line", "value", c(m_1, m_1.retro, m_1.lesioned, ideal_m_1, gt_m_1)) %>%
  group_by(M_line)

ggplot(
  df_M,
  aes(
    x = percept_number,
    y = value
  )
) + geom_line(aes(color = M_line)) + theme(aspect.ratio=1)