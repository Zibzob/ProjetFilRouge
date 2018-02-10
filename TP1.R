###########---------- TP1 - Flux de données -----------###########
##################################################################
##################################################################

rm(list = ls())

#install.packages("RCurl")
#install.packages("ggplot2")
library(RCurl)
library(tictoc)

#############################
############################# 

barhist <- function(mids,freq,col=NA,...)
{
  n=length(mids)
  d=diff(mids)
  d[1]=2*d[1]+d[2]
  d=c(d,2*d[n-1]+d[n-2])
  plot.new()
  plot.window(xlim=c(mids[1]-d[1]/2,mids[n]+d[n]/2),ylim=c(0,2*max(freq)))
  title(main = "Recursive hist and P-R kernel density estimate")
  legend(3.2, 0.6, legend = c("kernel", "N(0,1)"), col = c("red", "green2"),
         lty = 1, cex = 0.8)
  axis(1)
  axis(2)
  rect(mids-d/2,rep(0,length(mids)),mids+d/2,freq,col=col)
}

#############################
#############################

# URL <- getURL("http://crihan.airnormand.fr/IQA_commune.php?insee=76540")
# URL2 <- getURL("https://finance.google.com/finance?q=TICKER&output=json")

tic("Running time")
# Code complet intégrant l'ensemble des questions du TP1
vectURL <- rep(0,250)

# Première itération
vectURL[1] <- as.numeric(getURL("http://crihan.airnormand.fr/test.php"))

mean.rec <- vectURL[1]
min.rec <- vectURL[1]
max.rec <- vectURL[1]
sd.rec <- 0

delta <- 0.5
MIDS <- seq(-4, 4, by = delta)
FREQS <- rep(0, length(MIDS))

for(j in 1:length(MIDS)){
    FREQS[j] <- ifelse(vectURL[1] >= MIDS[j]-(delta/2) & vectURL[1] < MIDS[j]+(delta/2),1,0)
}

t <- seq(-4,4, by = 0.05)
gchap <- rep(0,length(t))

n <- 1


for(i in 2:length(vectURL)){
  
  vectURL[i] <- as.numeric(getURL("http://crihan.airnormand.fr/test.php"))
  
  sd.rec <- sqrt((n/(n+1))*sd.rec^2+(n/(n+1)^2)*(vectURL[i]-mean.rec)^2)
  mean.rec <- (n/(n+1))*mean.rec+(1/(n+1))*vectURL[i]
  min.rec <- min(min.rec, vectURL[i])
  max.rec <- max(max.rec, vectURL[i])
  
  cat(paste("val =", round(vectURL[i],3),"standard error = ",
            round(sd.rec,3), "mean = ", round(mean.rec,3),
            "min = ", round(min.rec,3), "max = ", round(max.rec,3))
      , sep = "\n")
  
  h <- sd.rec*((n+1)^(-0.2))
  
  for(j in 1:length(FREQS)){
    
    FREQS[j] <- FREQS[j] + (1/(n+1))*(ifelse(vectURL[i] >= MIDS[j]-(delta/2) & vectURL[i] < MIDS[j]+(delta/2),1,0) - FREQS[j])
    }
  
  
  for(k in 1:length(gchap)) {
  
    gchap[k] <- gchap[k] + (1/(n+1))*((1/h)*(exp(-((vectURL[i]-t[k])/h)^2/2)/(sqrt(2*pi)))-gchap[k])

  }
  
  
  barhist(MIDS,FREQS/rep(delta,length(MIDS)), col = "blue")
  lines(t,gchap, col="red", lwd = 2)
  lines(t,dnorm(t), col = "green2", lwd = 2)
  
  n <- n + 1
  
  #Q3 : ajout de la balise flush.console()
  #print(vectURL[i])
  #flush.console()
}
toc()




