Guys you need to install the dataset from
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000


Then unarchive it. There will be two folders with images. Combine them into one. Btw, each of the two folders may have duplicate images. Don't worry, just ignore duplicates when unarchiving. When you combine them, there will be no duplicates

Ideas for the project
1) split by patient id
2) more augmentation
3) stratified train-test splits
4) class-weighted CE loss
5) save the best model by validation metric, not on the last epoch
6) freeze everything except last 3-4 layers for transfer learning 
7) avoid data leakage when filling NA 
8) age is assumed to be 0-100. Dividing by 100 is fine but a smarter normalization could be used 
9) use more metrics like per-class F1 and ROC-AUC
10) use efficient net or deeper resnet

Making the project better
1) Visualize what the model “sees” - heatmaps on lesions, for instance. Professors love that. It turns an average model into an “AI-assisted diagnosis” story.
2) Make a web or some UI to make it look better
3) Try different configurations. Like another model, different number of frozen layers, different number of epochs, different dropout, etc


Building the model (hardest part)
1) Forget about metadata, classify images only
2) Build a data pipeline (transforms, dataset, dataloader)
3) build a model with resnet 50
4) add metadata info
5) make models easily switchable. For example can test in one line of code with resnet 50, efficient net, resnet 16
6) go to "making the project better"
