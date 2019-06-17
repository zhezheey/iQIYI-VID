### Training

- **get_gt.py**

	load the `train_gt.txt` and `val_gt.txt` and save as `gt.pickle`.

- **train.py**

	train the MLP model (*merge the training and validation set, then divide them randomly, 95% for training and 5% for validation*).

- **train.sh**

	script to train the models.

### Others
- **train_test.py**

	train the MLP model (*use the official training and validation set division*).

- **get_mAP.py**

	calculate the mAP value.
