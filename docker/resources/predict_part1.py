import argparse
import random
import pickle
import numpy as np
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


def get_data(feat_path):
    with open(feat_path, 'rb') as fin:
        feat_dict = pickle.load(fin, encoding='bytes')
    feat_keys = list(feat_dict.keys())
    feat_keys.sort()
    
    feat_list = []
    info_list = []
    count = 0
    
    # Part 1
    for video_name in feats_keys[:int(len(feats_keys) / 3)]:
        feats = feat_dict[video_name]
        if len(feats) == 0:
            continue
        count += 1
        
        # Keep the first half (if more than 8)
        feats.sort(key = lambda x: x[3], reverse=True)
        feats_keep = feats[:int(len(feats) / 2) if int(len(feats) / 4) >= 2 else int(len(feats))]
        for feat in feats_keep:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feat
            feat_list.append(feat_arr)
            info_list.append((video_name, qua_score))
            
    print('feat num:', len(feat_list))
    print('video num:', count)
    
    tmp = list(zip(feat_list, info_list))
    random.shuffle(tmp)
    feat_list, info_list = zip(*tmp)
    
    return np.array(feat_list), info_list


def get_result(info_list, result):
    result_dict = {}
    for i in range(len(info_list)):
        video_name = info_list[i][0]
        score = info_list[i][1]
        if video_name not in result_dict:
            result_dict[video_name] = [(result[i], score)]
        else:
            result_dict[video_name].append((result[i], score))
            
    # Use score as prediction weight
    output = []
    for video_name in result_dict:
        score_sum = 0
        final_result = np.zeros(10035, dtype=np.float32)
        for info in result_dict[video_name]:
            score = info[1] if info[1] > 0 else 1
            final_result += info[0] * score
            score_sum += score
        final_result /= score_sum
        output.append([video_name, final_result])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the results (part 1)')
    parser.add_argument('--feature', default='/data/materials/feat/face_test.pickle', help='feature (input)')
    parser.add_argument('--model', default='/data/model.hdf5', help='model (input)')
    parser.add_argument('--output', default='/data/part1.pickle', help='result (output)')
    args = parser.parse_args()
    
    random.seed(2019)
    
    feat_list, info_list = get_data(args.feature)
    print('Data preprocessed...')
    
    model = load_model(args.model)
    result = model.predict(feat_list, batch_size=256)
    print('Predicted...')
    
    output = get_result(info_list, result)
    result = None
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)
    print('Saved...')
