import os
import pickle

import pandas
from omegaconf import DictConfig

import hydra

BATCH_SIZE = 64
bg2label_dict = {
    'o': 0, # ocean
    'l': 0, # lake
    'f': 1, # forest
    'b': 1, # bamboo
}

@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    train_img_folder = cfg.root_dir + 'train/real/'
    train_img_files = [train_img_folder + f for f in os.listdir(train_img_folder) if f.endswith('.jpg')]
    train_test_metadata_csv = pandas.read_csv(cfg.root_dir + 'train_real_metadata.csv')
    train_test_metadata_csv['img_filename_no_path'] = train_test_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    gt_train_test = []
    bg_gt_train_test = []
    for img_name in train_img_files:
        img_name = img_name.replace(".jpg", "").split('/')[-1]
        gt_train_test.append(train_test_metadata_csv[train_test_metadata_csv['img_filename_no_path'] == img_name]['y'].values[0])
        bg = train_test_metadata_csv[train_test_metadata_csv['img_filename_no_path'] == img_name]['place_filename'].values[0].split('/')[1]
        bg_gt_train_test.append(bg2label_dict[bg])
    test_img_folder = cfg.root_dir + 'test/'
    test_img_files = [test_img_folder + f for f in os.listdir(test_img_folder) if f.endswith('.jpg')]
    Q_metadata_csv = pandas.read_csv(cfg.root_dir + 'test_metadata.csv')
    Q_metadata_csv['img_filename_no_path'] = Q_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    for img_name in test_img_files:
        img_name = img_name.replace(".jpg", "").split('/')[-1]
        gt_train_test.append(Q_metadata_csv[Q_metadata_csv['img_filename_no_path'] == img_name]['y'].values[0])
        bg = Q_metadata_csv[Q_metadata_csv['img_filename_no_path'] == img_name]['place_filename'].values[0].split('/')[1]
        bg_gt_train_test.append(bg2label_dict[bg])
    train_test_img_files = train_img_files + test_img_files
    numtrain_test = len(train_test_img_files)

    val_img_folder = cfg.root_dir + 'val/'
    val_img_files = [val_img_folder + f for f in os.listdir(val_img_folder) if f.endswith('.jpg')]
    gt_val = []
    bg_gt_val = []
    val_metadata_csv = pandas.read_csv(cfg.root_dir + 'val_metadata.csv')
    val_metadata_csv['img_filename_no_path'] = val_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1].replace(".jpg", ""))
    for img_name in val_img_files:
        img_name = img_name.replace(".jpg", "").split('/')[-1]
        gt_val.append(val_metadata_csv[val_metadata_csv['img_filename_no_path'] == img_name]['y'].values[0])
        bg = val_metadata_csv[val_metadata_csv['img_filename_no_path'] == img_name]['place_filename'].values[0].split('/')[1]
        bg_gt_val.append(bg2label_dict[bg])
    numQ = len(val_img_files)

    print("numtrain_test:", numtrain_test)
    print("numQ:", numQ)

    train_test_img_path_and_gt = [(f, gt) for f, gt in zip(train_test_img_files, gt_train_test)]
    train_test_img_path_and_bg_gt = [(f, gt) for f, gt in zip(train_test_img_files, bg_gt_train_test)]
    val_img_path_and_gt = [(f, gt) for f, gt in zip(val_img_files, gt_val)]
    val_img_path_and_bg_gt = [(f, gt) for f, gt in zip(val_img_files, bg_gt_val)]

    # save gt as pickle
    with open(cfg.save_dir + 'data/waterbird_gt_train_test.pkl', 'wb') as f:
        pickle.dump(train_test_img_path_and_gt, f)
    with open(cfg.save_dir + 'data/waterbird_bg_gt_train_test.pkl', 'wb') as f:
        pickle.dump(train_test_img_path_and_bg_gt, f)
    with open(cfg.save_dir + 'data/waterbird_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_gt, f)
    with open(cfg.save_dir + 'data/waterbird_bg_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_bg_gt, f)


if __name__ == "__main__":
    main()