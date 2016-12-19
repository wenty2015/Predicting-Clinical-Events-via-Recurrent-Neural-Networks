# Predicting-Clinical-Events-via-Recurrent-Neural-Networks

The project is to predict clinical events for a patient’s next visit, to assist the doctor and the patient to quickly retrieve useful information during the future visits. The dataset used to train the predictive model is a public dataset MIMIC-III[1], with historical patient events during the past several years. The input of the model is a sequence of the patient’s visits represented by standard diagnosis codes, and the output is the predicted codes in the next visit. The model used to solve the problem is based on GRU RNNs[2], which are efficient to be applied to sequences. Three modifications of the method are investigated to improve the prediction performance, including input embedding, auxiliary output training and additional features. Several baselines are implemented, which are evaluated by top-k recall. The GRU RNNs outperform the baselines, and results for modifications with different setting are compared.

The originial code for GRU RNNs in [2] is available on [3], and is modified in this project.

References
[1] https://mimic.physionet.org/
[2] Edward Choi, Mohammad Taha Bahadori, and Jimeng Sun. Doctor ai: Predicting clinical events via recurrent neural networks. arXiv preprint arXiv:1511.05942, 2015.
[3] https://github.com/mp2893/doctorai
