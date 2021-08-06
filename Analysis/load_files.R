library(tidyverse)
setwd("~/Documents/03_Yale/Projects/001_Mask_RCNN/ORB_project3/Data")

online = read_delim("online_output.csv", delim='&')
retrospective = read_delim("retrospective_output.csv", delim='&')
lesioned = read_delim("lesioned_output.csv", delim='&')
similarity = read_delim("similarity.csv", delim='&')
ideal_v_matrix = read_delim("ideal_v_matrix.csv", delim='&')
ground_truth_V = read_delim("ground_truth_V.csv", delim='&')

merged = merge(retrospective, lesioned, by = "video_number", suffixes = c(".retro",".lesion"))
