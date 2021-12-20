# Projekt zur Datenanalyse

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Result](#result)

## General info
This project was built during the module "Projekt zur Datenanalyse". 
The goal of the project was to build a system that recognizes jumps in jump diffusion data such as stock market data. The Isolation Forest and cutoff algorithms are used to detect jumps. The basis for this approach is the [master thesis by Carl Paulin](https://umu.diva-portal.org/smash/record.jsf?pid=diva2%3A1563784&dswid=-3478). In the first part of this project we wanted to reproduce the results of this master thesis.  
	
## Technologies
Project created with:
* Python 3.9
* Jupyter Notebooks

## Setup
The test data will be generated with the help of a Merton jump diffusion model. This process consists of a Brownian motion and Poisson distributed jumps. 
Then the features return, realized variance, realized bipower variation, difference, signed jumps will be extracted. 

### Extracted features

![alt text](https://github.com/Mastercheef/Projekt-Datenanalyse-/blob/main/Pictures/Testdata/Features_Testdata.png)


The Isolation Forest and cutoff method will then search for jumps in these features. 

### Isolation Forest and cutoff detecting on data

![alt text](https://github.com/Mastercheef/Projekt-Datenanalyse-/blob/main/Pictures/Testdata/MarkedJumps_Testdata.png)

To test the performance of these two algorithms, the F1-Score of each will be compared.
In the next step, the Isolation Forest and cutoff method will be applied to real financial data obtained through the Yahoo Finance API.

## Result
It was possible to generate data that follows a merton jump diffusion process. The features could be extracted and the Isolation Forest and cutoff method could be applied to the features to detect anomalies. The F1-Score, of the Isolation Forest in the original master thesis could be reproduced, but the resulting cutoff F1-Score was higher than in the original master thesis. 
