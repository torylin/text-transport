###################
###################
### Replication for the Hong Kong Experiment
###
### Fong and Grimmer "Causal Inference with Latent Treatments"
###
####################

## Set working directory to the folder which contains this file
## by using setwd("XXX") where XXX is the path to the file
## For example, on the author's computer, the command is
## setwd("D://Dropbox//SuperExp//LongPaper//dexp_replication")

## Load data from Dec 2019 experiment (d) and Oct 2020 Replication (drep)
d<- read.delim('HKData.csv', sep=',', row.names=NULL)
drep<- read.delim('HKRepData.csv', sep=',', row.names=NULL)

## Table 4
model1<- lm(resp~treatycommit + brave + evil + flag + threat + economy + treatyviolation, data = d)
model1rep <- lm(resp~treatycommit + brave + evil + flag + threat + economy + treatyviolation, data = drep)

summary(model1)
summary(model1rep)

## Two helper functions for the replication of the unmeasured dates treatment
## A function to loop over the texts and find specific words
find_words<- function(x, texts){
	output<- matrix(0, nrow = nrow(texts), ncol = 3)
	for(z in 1:3){
		for(y in 1:nrow(texts)){
			temp<- grep( x, tolower(as.character(texts[y,z])))
		output[y,z]<- ifelse(length(temp)==0, 0, 1)
	}
}
	return(output)
}

## A function to determine the observations with a specific version of a treatment
process_find<- function(mat, d){
	out<- rep(0, nrow(mat))
	for(z in 1:nrow(mat)){
		out[z]<- sum(mat[z,1:d$numtexts[z]])
	}
	return(out)

}

## Find words that mention specific dates (the identifying strings are 
## obtained from the construction of the texts)
texts <- d[,11:13]
d$time<- pmax(process_find(find_words('1997', texts), d), process_find(find_words('27', texts), d),
              process_find(find_words('1992', texts), d), process_find(find_words('1989', texts), d),
              process_find(find_words('20', texts), d))
## Test for violations
summary(lm(treatycommit ~ time, data = d))
summary(lm(resp ~ time, data = d))
summary(lm(resp~treatycommit + brave + evil + flag + threat + economy + treatyviolation + time, data = d))

## Repeat with October replication
textsrep <- drep[,11:13]
drep$time<- pmax(process_find(find_words('1997', textsrep), drep), process_find(find_words('27', textsrep), drep),
              process_find(find_words('1992', textsrep), drep), process_find(find_words('1989', textsrep), drep),
              process_find(find_words('20', textsrep), drep))
summary(lm(treatycommit ~ time, data = drep))
summary(lm(resp ~ time, data = drep))
summary(lm(resp~treatycommit + brave + evil + flag + threat + economy + treatyviolation + time, data = drep))

## Find the number of possible texts
## First load the number of texts in each of the possible latent treatments
ntreatycommit <- nrow(read.csv("HKarmstreatyobligation.csv", header = FALSE))
nbrave <- nrow(read.csv("HKarmsbrave.csv", header = FALSE))
nevil <- nrow(read.csv("HKarmsevil.csv", header = FALSE))
nflag <- nrow(read.csv("HKarmsflag.csv", header = FALSE))
nthreat <- nrow(read.csv("HKarmsthreat.csv", header = FALSE))
neconomy <- nrow(read.csv("HKarmseconomy.csv", header = FALSE))
ntreatyviolation <- nrow(read.csv("HKarmstreatyviolation.csv", header = FALSE))
controlarms <- c(nbrave,nevil,nflag,nthreat,neconomy,ntreatyviolation)

# Find the number of possible treated texts with two components
n2treat <- sum(ntreatycommit*controlarms)

# Find the number of possible control texts with two components
n2control <- 0
for(i in 1:(length(controlarms)-1)){
  for (j in (i+1):length(controlarms)){
    n2control <- n2control + controlarms[i]*controlarms[j]
  }
}

# Find the number of possible treated texts with three components
n3treat <- ntreatycommit*n2control

# Find the number of possible control texts with three components
n3control <- 0
for(i in 1:(length(controlarms)-2)){
  for (j in (i+1):(length(controlarms)-1)){
    for (k in (j+1):length(controlarms)){
      n3control <- n3control + controlarms[i]*controlarms[j]*controlarms[k]
    }
  }
}

# Find the number of possible texts overall
ntexts <- n2treat + n2control + n3treat + n3control
print(ntexts)