# ASX/ABS Early Warning Platform — Model Evaluation



## Purpose



This document summarises the current modelling approach, evaluation metrics, limitations, and next-stage improvements for the ASX/ABS Early Warning Platform.



The current implementation is an experimental financial-risk prototype and should not be interpreted as a production trading or credit-risk system.



---



# Current modelling approach



The system currently uses market-derived features built from ASX ticker data.



Example features include:



```text

ret\_12m

vol\_12m

drawdown\_12m

mom\_3m

liq\_proxy

