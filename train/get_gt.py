import pickle


gt_dict = {}

with open('../data/train_gt.txt') as ft:
    lines = ft.readlines()
    for line in lines:
        line = line.strip()
        gt_dict[line.split(' ')[0].split('.')[0]] = int(line.split(' ')[1])
        
with open('../data/val_gt.txt') as fv:
    lines = fv.readlines()
    for line in lines:
        line = line.strip()
        tmp = line.split(' ')
        for video in tmp[1:]:
            gt_dict[video.split('.')[0]] = int(tmp[0])
            
with open('../data/gt.pickle', 'wb') as fout:
    pickle.dump(gt_dict, fout)
    