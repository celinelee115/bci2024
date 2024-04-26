# BCI Competition of Finger Flexion Predictions Based on ECoG Recordings

University of Pennsylvania BE 521 Brain-Computer Interface, Spring 2024

Team: Van Leeuwen

Members: Celine Lee, Sabrina Weng

## Background

This final project aimed to predict finger flexion using intracranial EEG (ECoG) in three human subjects. The data and problem framing come from the 4th BCI Competition (Miller et al. 2008). This project proposed using ensemble modeling of XGBoost and AdaBoost for regression to train the preprocessed ECoG signal. The final correlation score achieved 0.4717 on the leaderboard. 

## Abstract 
The project focuses on predicting finger flexion movements using intracranial EEG recordings. Leveraging a dataset from the 4th International Brain-Computer Interfaces Competition, we aim to achieve a high correlation between predicted finger flexion movements and the target leaderboard data. Our final algorithm consists of dataset analysis, pre-processing of input data, model selection, and post-processing of predictions. In the pre-processing step, we apply a filter to filter out noises, extract relevant features, and calculate the response matrix on the raw data. In model selection, several models are trained and tested, ultimately culminating in the discovery that the ensemble model of AdaBoost and XGBoost outperforms all the others. Last but not least, the post-processing techniques further improve correlation by filtering out noises and interpolating predictions to match input size. Finally, utilizing our final algorithm, we achieve a correlation of 0.4717 between our finger flexion predictions and the target leaderboard data. 

## Flowchart
![flowchart2](https://github.com/celinelee115/bci2024/assets/123041751/8d4463c1-e2aa-457b-ad2d-f5e1c66cf925)
