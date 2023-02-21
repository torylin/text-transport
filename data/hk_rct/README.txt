Documentation of Replication Archive for "Causal Inference with Latent Treatments"

Codebook:
*Codebook.pdf: A codebook which describes the meaning of all of the variables used in both studies.

Data:
*HKarmsbrave.csv: The texts used for the brave latent treatment in the Honk Kong study.
*HKarmseconomy.csv: The texts used for the economy latent treatment in the Hong Kong study.
*HKarmsevil.csv: The texts used for the evil latent treatment in the Hong Kong study.
*HKarmsflag.csv: The texts used for the flag latent treatment in the Hong Kong study.
*HKarmsthreat.csv: The texts used for the threat latent treatment in the Hong Kong study.
*HKarmstreatyobligation.csv: The texts used for the treaty commitment latent treatment in the Hong Kong study.
*HKarmstreatyviolation.csv: the texts used for the violation latent treatment in the Hong Kong study.
*HKData.csv: The data from the first survey experiment for the Hong Kong study from 5.1, conducted in December, 2019.
*HKRepData.csv: The data from the replication of the Hong Kong Study from 5.1, conducted in October, 2020.
*trumpdt.csv: The data to replicate the Twitter study from 5.2

Code:
*HK_Replication.R: code to replicate the Hong Kong study from 5.1, along with all associated tables and figures.
*Trump_Replication.R: code to replicate the Twitter study from 5.2, along with all associated tables and figures.
					  Only runs properly on versions of R from before 3.6 due to changes to how the sample()
					  function in base R draws from random seeds.
*Download_Yougov.py: code to download the data from YouGov for running the Trump Tweet analysis.  It is not necessary
					 to run this code to replicate the results (all necessary data is contained in trumpdt.csv),
					 but it permits the further exploration of the YouGov data if required.  Only runs on Python 2.7.

Software:
*R 3.5.2
*Packages:
**car 3.0-10
**texteffect 0.3
**tidytext 0.3.0

Data Sources:
*The Hong Kong study draws data from two original survey experiments.
*The study of President Trump's tweets relies on data from YouGov's Tweet Index Citation: YouGov. It includes all tweets from 2/4/2017-10/31/2017.  "YouGov Tweet Index."  tweetindex.yougov.com.   Accessed 11/12/2017.  
