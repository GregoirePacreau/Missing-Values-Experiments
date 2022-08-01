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

parsList = c(FALSE, 5, 0.15, 1e-12, TRUE)

DIdata = DI(data, initEst, crit, maxits, quant, maxCol, checkPars=parsList)

wname = sub('.csv', '', args[1])
write.csv(DIdata$center, paste(wname, '_res_mu.csv', sep=''), row.names=FALSE)
write.csv(DIdata$cov, paste(wname, '_res_S.csv', sep=''), row.names=FALSE)