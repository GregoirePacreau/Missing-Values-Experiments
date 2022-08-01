library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
quant = as.double(args[2])

if ncol(data) < 250 {
    parsList = c(0.5, 3, 1e-12, 'automatic', 0.99, 0.5, 'wmean', FALSE, TRUE, 25000, FALSE, '1stepM', 'gkwls', 'wrap', 100)

}
else{
    parsList = c(0.5, 3, 1e-12, 'automatic', 0.99, 0.5, 'wmean', FALSE, TRUE, 25000, TRUE, '1stepM', 'gkwls', 'wrap', 100)

}
DDCdata = DDC(data, DDCpars=parsList)

# Outlier test using chi-squared quantiles
isOutlier = DDCdata$stdResid > sqrt(qchisq(quant, 1))

wname = sub('.csv', '', args[1])
write.csv(isOutlier, paste(wname, '_res.csv', sep=''), row.names=FALSE)
