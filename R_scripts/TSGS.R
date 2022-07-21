library(GSE)

# arguments are:
# - location of the data (in str)
# - quantile for outlier test (int, in percent)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])
filter = args[2]
partial_inpute = args[3]
tol = as.double(args[4])
maxiter = strtoi(args[5])
method = args[6]
init = args[7]
params = c(data, filter, partial_inpute, tol, maxiter,method, init)
if (len(args) > 7){
  mu0 = as.double(args[8])
  params.append(mu0)
}
if (len(args) > 8){
  S0 = as.matrix(args[9])
  params.append(S0)
}


TSGSdata = do.call(TSGS, params)

wname = sub('.csv', '', args[1])
write.csv(c(TSGSdata$mu, TSGSdata$S, TSGSdata$getFiltDat), paste(wname, '_res.csv', sep=''))
