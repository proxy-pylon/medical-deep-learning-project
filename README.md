# medical-deep-learning-project

## Introduction

This project uses deep learning methods to detect skin cancer.

## Data we will likely use:

- Dermoscopic image datasets. Some skin cancer images are openly available on the internet. We can access them to train the model
- Optional clinical photos. Some real-world samples for evaluation of our model
- Metadata like age, gender, etc. Seeing a photo of a hand is good but it's better to know that this is a hand of an old guy who is likely at risk. We can easily gain more accuracy by using this metadata for our predictions

## Possible problems and solutions

- Lack of variability in dataset. We might get cancer images mostly for females, mostly for aged people, mostly for Europeans, etc. There can be very few instances of dark-skinned people or kids for example. So the classifier may be not sensitive for darker skin tones.
- Melanoma is rare. Our data is likely to be very imbalanced. The classifier can just always say "no melanoma here" and be correct 99% of the time. Solutions:
- - Search for more data in the internet. Search  English and Russian sources for more images.
  - More penalty in the loss function for the cancer instances 
  - What else?
- One patient may have multiple photos of his cancer. So if one patient's images are both in the train and test, we have a problem of data leakage. Solution is to split train and test by patient ID
- Images are going to be from different angles and lighting. Solution is data augmentation, color normalization

## Short-term plans 

- Ask prof for permission for this topic
- If you don't have any experience with computer vision (CNNs), it's a good time to start in advance
- Explore databases to find data
- Read papers and explore github because they will cite the datasets that they used and we can use it too
- Decide where we will store our data. I guess we need a huggingface repository? We need a consistent storage format for all the different sources. So even if we mash together different datasets, the format of data should be uniform. I guess we will use some CSV with columns or JSONs and a separate folder for images
- Run a pre-trained ResNet on one of the datasets to have a baseline. Everything we come up with should beat this baseline

## Minor details

- Fix seed number for reproducibility

## Metrics

We need to decide on the metrics. 
I guess the list is the following
- ROC-AUC. Baza
- PR-AUC. Baza
- Confusion matrix. Baza
- "Sensitivity at fixed specificity (e.g. sens@95% spec, sens@90% spec)". ChatGPT suggests this but I don't even know what that is
- Other metrics for fun

## Conclusion

In the end we should get a good, rich, and diverse dataset with many examples of cancer. We should have a model that scores well on the metrics mentioned above. We can use these metrics to write a report. The more metrics we have, the easier, faster, and better the report will be.
