library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
quant = strtoi(args[2])/100

DDCdata = DDC(data)

# Outlier test using chi-squared quantiles
isOutlier = DDCdata$stdResid > sqrt(qchisq(quant, 1))

wname = sub('.csv', '', args[1])
write.csv(isOutlier, paste(wname, '_res.csv', sep=''))
