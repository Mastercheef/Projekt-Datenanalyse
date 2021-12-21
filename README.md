# Data Analysis Project (H-BRS)
___

### <center> Anomaly Detection in Financial Market Data
####  <center>  using Isolation Forest and Features
___

&nbsp;

**Authors:** &nbsp;Erik Autenrieth, Pierre Zimmermann &nbsp;

**Date:** &nbsp;&nbsp;&nbsp;&nbsp;  20.12.2021
___

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Result](#result)

## General info

Anomalies can occur in financial time series and cause a sudden price jump.
The anomaly detection in financial market data is important for modeling and improving analytical and predictive models as well as detecting market manipulation.

These jumps can be simulated in a Merton jump-diffusion process, which can model a financial time series.  The test data marked in this way can then be examined for outliers using machine learning algorithms.
The Isolation Forest was tested with various features on this generated batch data to be able to apply the model to real financial data.

Here, above all, concepts and results from the master thesis 
[”Detecting anomalies in data streams driven by a jump-diffusion process”](https://umu.diva-portal.org/smash/record.jsf?pid=diva2%3A1563784&dswid=-3478) by Carl Paulin, were used to reproduce the results and to be able to use the resulting models for the analysis of real data.

While the master thesis uses several algorithms such as Robust Random Cut Forest and Isolation Forest for streaming data, this work is limited to reproducing the results of the Isolation Forest on the same features such as the returns, realized variation, realized bipower variation, and realized semi-variation. The methods were also tested on six different jump rates which served as contamination for the Isolation Forest.

The results of the examined thesis can be partially confirmed. This states that the results of the Isolation Forest get better when features are used, whereas when three features are used, the results get worse. The results of this work show that all features deliver relatively similar result. However, it shows that feature extraction is worthwhile, especially since anomalies are rarely detected without using a feature.
While the CutOff in the master thesis was always worse than the Isolation Forest, this was better in the analysis of this work.
However, this has no effect on the analysis of the real data, as it does not contain any signed jumps, which are necessary for calculating a CutOff.

## Technologies
Project created with:
* Python 3.9
* Jupyter Notebooks

## Setup
The test data will be generated with the help of a Merton jump diffusion model. This process consists of a Brownian motion and Poisson distributed jumps. 
Then the features return, realized variance, realized bipower variation, difference, signed jumps will be extracted. 

### Extracted Features

![alt text](https://github.com/Mastercheef/Projekt-Datenanalyse/blob/main/Pictures/Testdata/Features_Testdata.png)


The Isolation Forest and CutOff method will then search for jumps in these features. 

### Isolation Forest and CutOff Detecting on Data

![alt text](https://github.com/Mastercheef/Projekt-Datenanalyse/blob/main/Pictures/Testdata/MarkedJumps_Testdata.png)

To test the performance of these two algorithms, the F1-Score of each will be compared.
In the next step, the Isolation Forest and cutoff method will be applied to real financial data obtained through the Yahoo Finance API.

## Result
It was possible to generate data that follows a Merton jump diffusion process. The features could be extracted and the Isolation Forest and cutoff method could be applied to the features to detect anomalies. The F1-Score, of the Isolation Forest in the original master thesis could be reproduced, but the resulting CutOff F1-Score was higher than in the original master thesis. 
