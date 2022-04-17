# GBERT
Implementation of https://arxiv.org/pdf/1906.00346.pdf for CS598 Project

## PRESCRIPTIONS.CSV and DIAGNOSIS_ICD.csv 
These are referenced in the code base but we have not checked in since this database is granted access via physionet

## Assumptions:
1. Single visit: We have assumed it to be for now single hospital admission
2. % threshold for most-frequent codes: _____ (impacting vocab size)
3. Mapping files to use: [https://github.com/sjy1203/GAMENet/blob/master/data/ndc2rxnorm_mapping.txt](NDC2RxNorm) and [https://github.com/sjy1203/GAMENet/blob/master/data/ndc2atc_level4.csv](RxNorm2ATC4)

## Environment setup:
We have run this code on:
1. python: 3.9.12
2. pytorch: 1.11.0
3. pytorch geometric: 2.0.4
All using anaconda.

## Code checkin instructions:
1. Maintain our local colab notebooks/python files (This would help in debugging with the help of other person)
2. Checkin after raising a PR
3. Do squash and merge


