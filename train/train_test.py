import argparse
import random
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)
KTF.set_session(session)


def get_data(feat_path, gt_path, threshold):
    with open(feat_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    with open(gt_path, 'rb') as fin:
        gt_dict = pickle.load(fin, encoding='iso-8859-1')
    threshold = [int(d) for d in args.threshold.split(',')]
    
    x = []
    y = []
    
    for video_name in feats_dict:
        feats = feats_dict[video_name]
        label = gt_dict[video_name] - 1 if video_name in gt_dict else 10034
        if len(feats) == 0:
            continue
        feats.sort(key = lambda x: x[3], reverse=True)
        
        # threshold to filter the features
        for feat in feats:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feat
            if qua_score > threshold[0] and qua_score < threshold[1]:
                x.append(feat_arr)
                y.append(label)
                
    return np.array(x), np.array(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model using face features')
    parser.add_argument('--train', default='../feat/face_train_v2.pickle', help='face features of training set (input)')
    parser.add_argument('--val', default='../feat/face_val_v2.pickle', help='face features of validation set (input)')
    parser.add_argument('--gt', default='../data/gt.pickle', help='ground truth of training and validation set (input)')
    parser.add_argument('--model', default='model/model.hdf5', help='directory to save model (output)')
    parser.add_argument('--seed', type=int, default=2019, help='random seed')
    parser.add_argument('--threshold', default='40,200', help='threshold of score')
    parser.add_argument('--epoch', type=int, default=5, help='training epoch')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # load training data
    train_x, train_y = get_data(args.train, args.gt, args.threshold)
    val_x, val_y = get_data(args.val, args.gt, args.threshold)
    print('training feat shape:', train_x.shape)
    print('validation feat shape:', val_x.shape)
    
    # MLP model
    # TODO(zheey): Deeper model can achieve better results.
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_dim=512))
    model.add(BatchNormalization())
    model.add(Dropout(0.9))
    model.add(Dense(10035, activation='softmax'))
    model.compile(optimizer=optimizers.adam(lr=0.0008), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_x, y=train_y, epochs=args.epoch, batch_size=32768, validation_data=(val_x, val_y), shuffle=True)
    model.save(args.model)
    