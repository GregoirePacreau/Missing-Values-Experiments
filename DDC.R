library(cellWise)

# arguments are:
# - location of the data (in str)
args = commandArgs(trailingOnly=TRUE)
data = read.csv(args[1])

DDCdata = DDC(data)

wname = sub('\\.csv$', '', args[1])
write.csv(paste(wname, '_res.csv'))
