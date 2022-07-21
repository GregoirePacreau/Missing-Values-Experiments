library(cellWise)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
initEst = args[2]
crit = as.double(args[3])
maxits = as.integer(args[4])
quant = as.double(args[5])
maxCol = as.double(args[6])
params = c(data, initEst, crit, maxits, quant, maxCol)

DIdata = call.do(DI, params)

res = c(DIdata$center, DIdata$cov)

wname = sub('.csv', '', args[1])
write.csv(res, paste(wname, '_res.csv', sep=''))
