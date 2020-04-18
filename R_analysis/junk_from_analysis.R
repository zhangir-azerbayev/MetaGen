# #for what portion does MH get the mode reality right? How about PF?
# nrows = dim(data)[1]
# total_MH = 0
# total_PF = 0
# for(i in 1:nrows){
#   if(identical(data$mode.realities.MH[i],data$gt_R[i])){
#     total_MH = total_MH+1
#   }
#   if(identical(data$mode.realities.PF[i],data$gt_R[i])){
#     total_PF = total_PF+1
#   }
# }
# proportion_MH = total_MH / nrows
# proportion_PF = total_PF / nrows
# proportion_MH
# proportion_PF


# #Average distance between average reality and the real one? MH and PF?
# euc.mh <- as.character(data$Euclidean.distance.between.avg_Rs.and.gt_R.MH)
# euc.mh <- unlist(strsplit(euc.mh, split = ","))
# mean(as.numeric(euc.mh))
# 
# euc.pf <- as.character(data$Euclidean.distance.between.avg_Rs.and.gt_R.PF)
# euc.pf <- unlist(strsplit(euc.pf, split = ","))
# mean(as.numeric(euc.pf))
# 
# 
# #Average distance on FA and M? compared to mean?
# mean(data$Euclidean.distance.FA.MH)
# mean(data$Euclidean.distance.M.MH)
# mean(data$Euclidean.distance.FA.PF)
# #should be M.PF
# mean(data$Euclidean.distance.M.PF)
# 
# mean(data$Avg.Euclidean.distance.FA.between.expectation.and.gt_V)
# #should be gt_V
# mean(data$Avg.Euclidean.distance.M.between.expectation.and.gt_V)