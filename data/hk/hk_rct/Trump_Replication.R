###################
###################
### Replication for Trump Tweet Analysis
###
### Fong and Grimmer "Causal Inference with Latent Treatments"
###
####################

## To replicate the results, it is essential to run this script in versions of R from
## before R3.6.  R 3.6 changed how the sample() function draws from random seeds, 
## which leads to completely different results.  The original analysis was conducted using
## R 3.5.2.

install.packages("tidytext")
install.packages("texteffect")
install.packages("textdata")
install.packages("car")
install.packages("xtable")

library(tidytext)
library(texteffect)
library(textdata)
library(car)
library(xtable)

## Set working directory to the folder which contains this file
## by using setwd("XXX") where XXX is the path to the file
## For example, on the author's computer, the command is
## setwd("D://Dropbox//SuperExp//LongPaper//dexp_replication")

dat <- read.csv("trumpdt.csv")

## Divide into outcome (Y), party ID indicator (G), and 
## word counts (X)
Y <- dat[,1]
G <- dat[,2:4]
X <- dat[,5:ncol(dat)]

rm(dat)

## Divide into training and test sets, ensuring that
## the Republican, Democrat, and Independent observations
## for any given tweet either all go into the training set
## or all go into the test set
set.seed(12082017)
training.tweets <- sample(1:(nrow(X)/3), nrow(X)/3*.5)
train.ind <- c()
for (i in 1:length(training.tweets)){
  train.ind <- c(train.ind, 3*(training.tweets[i]-1)+(1:3))
}

## Fit sIBP with many different parameter figurations so the analyst can choose
## the most substantively interesting run
## Note: This will take a while to run (approx 20 minutes)
sibp.search <- sibp_param_search(X, Y, K = 5, alphas = c(2,3,4), sigmasq.ns = c(0.5, 0.75, 1), 
                                 iters = 5, train.ind = train.ind, G = G, seed = 12082017)

## The analysts chose this as the most substantively interesting
## (and selected it before analyzing the test set, as prescribed
## in Fong and Grimmer, 2016)
sibp.fit <- sibp.search[["3"]][["0.5"]][[1]]

## Table 5
xtable(sibp_top_words(sibp.fit, colnames(X), verbose = TRUE))

## Infer values of latent treatments in the test-set
set.seed(12092017)
X.test <- t(apply(X[sibp.fit$test.ind,], 1, function(x) (x - sibp.fit$meanX)/sibp.fit$sdX))
nu.test <- infer_Z(sibp.fit, X)
Z.train <- matrix(as.numeric(sibp.fit$nu >= 0.5), ncol = 5)
Z.test <- matrix(as.numeric(nu.test >= 0.5), ncol = 5)

## Construct data for anaysis split out by partisanship
dat2 <- data.frame(Y[sibp.fit$test.ind], G[sibp.fit$test.ind,])
colnames(dat2) <- c("Y", "ind", "dem", "rep")
dat2$Z1 <- Z.test[,1]
dat2$Z2 <- Z.test[,2]
dat2$Z3 <- Z.test[,3]
dat2$Z4 <- Z.test[,4]
dat2$Z5 <- Z.test[,5]

## Get sentiment scores via AFINN and dichotomize into positive or negative
start<- as.matrix(get_sentiments('afinn'))
use0 <- match(colnames(X), start[,1])
use <- use0[which(!is.na(use0))]
use_col<- which(is.na(match(colnames(X), start[,1]))==F)
sents<- as.matrix(X[sibp.fit$test.ind,use_col])%*%as.numeric(start[use,2])
dat2$sents <- I(sents > 0)

## Table 6, first leaving sentiment unmeasured then measuring positive sentiment
summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$dem == 1)))
summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$ind == 1)))
summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$rep == 1)))

summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5 + sents, data = dat2, subset = which(dat2$dem == 1)))
summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5 + sents, data = dat2, subset = which(dat2$ind == 1)))
summary(lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5 + sents, data = dat2, subset = which(dat2$rep == 1)))

## Table 7, can choose any party because all see the texts
## with the same frequency
summary(lm(sents ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset=which(dat2$rep == 1)))

fit.rep <- lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$rep == 1))
fit.dem <- lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$dem == 1))
fit.ind <- lm(Y ~ Z1 + Z2 + Z3 + Z4 + Z5, data = dat2, subset = which(dat2$ind == 1))

## Appendix I1
apply(sibp.fit$nu, 2, order, decreasing = TRUE)[1:9,]
apply(sibp.fit$nu, 2, sort, decreasing = TRUE)[1:9,]

## Appendix I2
## Note: would be the same looking at Democrats or Independents
with(dat2[which(dat2$rep == 1),], cor(cbind(Z1,Z2,Z3,Z4,Z5)))

## Appendix I3
## Replicate with first order interactions
fit.dem.int<- lm(Y~Z1*Z2 + Z1*Z3 + Z1*Z4 + Z1*Z5 + Z2*Z3 + Z2*Z4 + Z2*Z5 + Z3*Z4 + Z3*Z5 + Z4*Z5, data = dat2, subset=which(dat2$dem==1))
fit.ind.int<- lm(Y~Z1*Z2 + Z1*Z3 + Z1*Z4 + Z1*Z5 + Z2*Z3 + Z2*Z4 + Z2*Z5 + Z3*Z4 + Z3*Z5 + Z4*Z5, data = dat2, subset=which(dat2$ind==1))
fit.rep.int<- lm(Y~Z1*Z2 + Z1*Z3 + Z1*Z4 + Z1*Z5 + Z2*Z3 + Z2*Z4 + Z2*Z5 + Z3*Z4 + Z3*Z5 + Z4*Z5, data = dat2, subset=which(dat2$rep==1))

summary(fit.dem.int)
summary(fit.ind.int)
summary(fit.rep.int)