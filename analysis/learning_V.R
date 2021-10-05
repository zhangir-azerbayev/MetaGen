library(tidyverse)
#setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/Data/20particles_threshold64_18945877")
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/scratch_work_07_16_21/09_30/dataset_1_obj_model2/detr/")

online = read_delim("shuffle_3_detr/online_V.csv", delim='&')
online$order_run = as.numeric(online$order_run)
retrospective = read_delim("shuffle_3_detr/retro_V.csv", delim='&') #has a bunch of extra columns but oh well
retrospective$order_run= as.numeric(retrospective$order_run)
#lesioned = read_delim("lesioned_V.csv", delim='&')
#lesioned$video_number = as.numeric(lesioned$video_number)
#similarity = read_delim("similarity.csv", delim='&')
#ideal_v_matrix = read_delim("ideal_v_matrix.csv", delim='&')
ground_truth_V = read_delim("ground_truth_V.csv", delim='&')
online = read_csv("shuffle_3_detr/online_V_processed.csv")


merged1 = merge(retrospective, online, by = "order_run", suffixes = c(".retro",".online"))
#merged2 = merge(online, merged1, by = "video_number")
#merged3 = merge(merged2, ideal_v_matrix, by = "video_number")
to_merge <- ground_truth_V %>% select(-video_number, -X12)
merged4 = cbind(merged1, ground_truth_V)

#to graph every model's version of m_1
# df_M = gather(merged4, "M_line", "value", c(m_1, m_1.retro, m_1.lesioned, ideal_m_1, gt_m_1)) %>%
#   group_by(M_line)
df_M = gather(merged4, "M_line", "value", c(m_5, ideal_m_5, gt_m_5, m_5.lesion)) %>%
  group_by(M_line)

ggplot(
  df_M,
  aes(
    x = video_number,
    y = value
  )
) + geom_line(aes(color = M_line)) + ylim(0,1)

df_FA = gather(merged4, "FA_line", "value", c(fa_1, ideal_fa_1, gt_fa_1, fa_1.lesion)) %>%
  group_by(FA_line)

df_FA$value = as.numeric(df_FA$value)

ggplot(
  df_FA,
  aes(
    x = video_number,
    y = value
  )
) + geom_line(aes(color = FA_line))  + ylim(0,1)
  # + theme(aspect.ratio=1)


View(merged4 %>% select(inferred_mode_realities, ground_truth_world_states))

############################################################
#lesioned_MSE = ((1 - ground_truth_V$gt_fa_1[1])^2 + (1 - ground_truth_V$gt_fa_2[1])^2 + (1 - ground_truth_V$gt_fa_3[1])^2 + (1 - ground_truth_V$gt_fa_4[1])^2 + (1 - ground_truth_V$gt_fa_5[1])^2 + (0.5 - ground_truth_V$gt_m_1[1])^2 + (0.5 - ground_truth_V$gt_m_2[1])^2 + (0.5 - ground_truth_V$gt_m_3[1])^2 + (0.5 - ground_truth_V$gt_m_4[1])^2 + (0.5 - ground_truth_V$gt_m_5[1])^2)/10
#MSE Plot
#MSE Plot
ggplot(
  online,
  aes(
    x = order_run,
    y = MSE,
    ymin = lower_MSE,
    ymax = upper_MSE,
  )
) + geom_line() + geom_ribbon(alpha = 0.5)# + 
  #geom_hline(yintercept = lesioned_MSE) + 
  #ylim(c(0,0.55))

############################################################
#show learning V averaged across runs
ground_truth_V = read_delim("ground_truth_V.csv", delim='&')

online_0 = read_csv("shuffle_0_detr/online_V_processed.csv")
online_1 = read_csv("shuffle_1_detr/online_V_processed.csv")
online_2 = read_csv("shuffle_2_detr/online_V_processed.csv")
online_3 = read_csv("shuffle_3_detr/online_V_processed.csv")

merged1 = merge(online_0, online_1, by = c("order_run"), suffixes = c(".0",".1"))
merged2 = merge(online_2, online_3, by = c("order_run"), suffixes = c(".2",".3"))
merged = merge(merged1, merged2, by = c("order_run"))

df <- merged %>% mutate(MSE = (MSE.0 + MSE.1 + MSE.2 + MSE.3)/4,
                        upper = (upper_MSE.0 + upper_MSE.1 + upper_MSE.2 + upper_MSE.3)/4,
                        lower = (lower_MSE.0 + lower_MSE.1 + lower_MSE.2 + lower_MSE.3)/4)

ggplot(
  df,
  aes(
    x = order_run,
    y = MSE,
    ymin = lower,
    ymax = upper,
  )
) + geom_line() + geom_ribbon(alpha = 0.5) +
  theme(aspect.ratio = 1)

