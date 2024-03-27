import pickle

import pandas
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    train_img_folder = cfg.root_dir + 'train/real/'
    train_metadata_csv = pandas.read_csv(cfg.root_dir + 'train_real_metadata.csv')
    train_metadata_csv['img_filename_no_path'] = train_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    train_test_img_files = []
    gt_train_test = []
    bg_gt_train_test = []
    for _, row in train_metadata_csv.iterrows():
        train_test_img_files.append(train_img_folder + row['img_filename_no_path'])
        gt_train_test.append(row['y'])
        bg_gt_train_test.append(row["place"])
    
    test_img_folder = cfg.root_dir + 'test/'
    test_metadata_csv = pandas.read_csv(cfg.root_dir + 'test_metadata.csv')
    test_metadata_csv['img_filename_no_path'] = test_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    for _, row in test_metadata_csv.iterrows():
        train_test_img_files.append(test_img_folder + row['img_filename_no_path'])
        gt_train_test.append(row['y'])
        bg_gt_train_test.append(row["place"])
    numtrain_test = len(train_test_img_files)

    val_img_folder = cfg.root_dir + 'val/'
    val_metadata_csv = pandas.read_csv(cfg.root_dir + 'val_metadata.csv')
    val_metadata_csv['img_filename_no_path'] = val_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    val_img_files = []
    gt_val = []
    bg_gt_val = []
    for _, row in val_metadata_csv.iterrows():
        val_img_files.append(val_img_folder + row['img_filename_no_path'])
        gt_val.append(row['y'])
        bg_gt_train_test.append(row["place"])
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