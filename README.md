# Pre-training of Graph Augmented Transformers(G-BERT) for Medication Recommendation
Implementation of https://arxiv.org/pdf/1906.00346.pdf for CS598 Project

## Requirements:
Refer Conda online documentation to setup environment using env.yml checked-in in the repo.

## Pre-Training/Training/Evaluation:
Refer Part 11 of main.ipynb and run the ```main()``` method

## Pre-trained and Ablation models:
We have checked in both pre-trained and ablation models for easy down-stream task evaluation. They are present in ```./Models```

## Results:
Results are checked in Part 11 of main.ipynb. 

## Details about Our implementation:
### MIMIC-III
Tables like PRESCRIPTIONS.CSV and DIAGNOSIS_ICD.csv which are present in MIMIC-III data-set are referenced in the code base but we have not checked in since this data-set is granted access via physionet

### Assumptions:
1. Mapping files to use: [https://github.com/sjy1203/GAMENet/blob/master/data/ndc2rxnorm_mapping.txt](NDC2RxNorm) and [https://github.com/sjy1203/GAMENet/blob/master/data/ndc2atc_level4.csv](RxNorm2ATC4). Reason to use old mapping files is mentioned in Part-1 of main.ipynb and is to prevent loss of 25% ATC4 codes
2. Single visit patient: We have assumed it to be for now patient with single hospital admission. Paper also talks in this direction. Moreover, we have taken whole admission data and not only first 24 data.
3. Max seq len input to G-BERT: for pre-training it is 100(+1 for CLS) for ATC4 codes and 39(+1 for CLS) for ICD codes. And for training its 92+1 and 39+1 respectively. This is direct consequence of taking whole hospital visit data.  
4. Number of most-frequent ICD9 codes: most frequent 2000 ICD9 codes are considered(impacting vocab size, and hence parameter size). We have mentioned analysis on this in Part-1 of main.ipynb and i.e. only 3% visits had >30% "[UNK]" token when top 2000 diagnosis codes are considered
5. Paper talks about masking in confusing manner. It says we mask 15% of codes, but we are not masking fixed 15% of codes. Rather we are masking codes with probability of 15%. This might mean that at time we mask more than or less than 15% codes in a sequence. However over large iterations we would have ~15% tokens being masked.
6. Direct consequence of (4.) and (5.): We face "[UNK]" token while preparing pre-training dataset and masking. Current masking strategy is to mask "[UNK]" token as well. This was adapted since had the token be known it would have been masked. Moreover this does not affect the downstream pre-training objectives.
7. Hyper-parameter choice:
    <ul>
        <li>Learning rate: 0.001 though we have seen that choosing 5e-4 has faster learning (maybe due to less gradient missteps)</li>
        <li>threshold for marking probability as 1 : 0.5 (though paper talks about 0.3, and need some more test runs to check this)</li>
    </ul>
8. Logistic Regression: Since it is not clear that we have to sum up previous visits or not. We are going ahead with not summing up previous visits. Also breakdown of training and test data not given so assuming it to be 0.8:0.2

### Doubts in Author implementation:
1. Author implementation has two vocabularies but ideally BERT style pre-training should have one global vocab
2. [<i><a href="https://github.com/jshang123/G-Bert">G-BERT author's implementation</a></i> takes only first 24 hrs data of a single visit patient though paper talks about hospital visit/admission whole.
3. As a result of (2.) author implementation have max seq len (input to G-BERT) as 62 ATC4 in pre-training. This is direct consequence of taking less data from visit (first 24 hrs)
4. Some SUBJECT_ID in author provided <a href="https://github.com/jshang123/G-Bert/blob/master/data/data-single-visit.pkl">data set</a> for single visit has missing Subject Ids like: 11, 86, 92. Though there is no clear reason why? This might be hidden due to complex data processing done by author in his/her git repo.
5. Further expanding on (4.): Even when author is filtering top 2k diagnosis codes based on frequency, we could not find any extra token in the <a href="https://github.com/jshang123/G-Bert/blob/master/data/data-single-visit.pkl">data set</a>, that would later qualify for "[UNK]". This points that author has done some more dropping of SUBJECT_IDs based on vocab of ICD in consideration. 

## Contributing:
1. Maintain a local notebooks/python files (This would help in debugging with the help of other person)
2. Check-in after raising a PR
3. Do squash and merge


