#simply want to print inference about miss rate for 
#category 2 and fa rate for category 5

miss2 <- vector(mode="double", length=n_percepts)
fa5 <- vector(mode="double", length=n_percepts)
for(p in 1:n_percepts){
  #temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1] #[-1] is needed sometimes because there might be a space at the beginning of gt_V
  temp_var <- unlist(strsplit(as.character(data[[list[p]]]), split = " "))[-1]
  mat <- matrix(as.numeric(temp_var), ncol=2, byrow=TRUE)
  miss2[p] <- mat[2,2]
  fa5[p] <- mat[5,1]
}

gt_miss2 <- rep(0.0, n_percepts)
gt_fa5 <- rep(0.1, n_percepts)

chance_miss2 <- rep(0.5, n_percepts)
chance_fa5 <- rep(0.5, n_percepts)

percept_number <- rep(1:n_percepts)

toPlot <- data.frame(percept_number, miss2, fa5, gt_miss2, gt_fa5, chance_miss2, chance_fa5)

df_M = gather(toPlot, "M_line", "value", c(miss2, gt_miss2)) %>%
  group_by(M_line)

ggplot(
  df_M,
  aes(
    x = percept_number,
    y = value
  )
) + geom_line(aes(color = M_line)) + theme(aspect.ratio=1)


df_FA = gather(toPlot, "FA_line", "value", c(fa5, gt_fa5)) %>%
  group_by(FA_line)

ggplot(
  df_FA,
  aes(
    x = percept_number,
    y = value
  )
) + geom_line(aes(color = FA_line)) + theme(aspect.ratio=1)
