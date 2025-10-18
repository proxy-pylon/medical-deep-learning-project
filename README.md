Guys you need to install the dataset from
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


Then unarchive it. There will be two folders with images. Combine them into one. Btw, each of the two folders may have duplicate images. Don't worry, just ignore duplicates when unarchiving. When you combine them, there will be no duplicates

To solve class imbalance:
1. Use ROC-AUC or F1 to select the best threshold. Don't save by accuracy
2. Adjust class weights. Double or triple it
3. Oversample the minority class
4. Use binary focal loss instead of cross-entropy loss
