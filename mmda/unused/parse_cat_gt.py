import pickle

import pandas
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    metadata_csv = pandas.read_csv(cfg.root95_dir + 'metadata.csv')
    metadata_csv['img_filename_no_path'] = metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    train_rows = []
    test_rows = []
    val_rows = []
    for _, row in metadata_csv.iterrows():
        if row['split'] == 0:
            train_rows.append(row)
        elif row['split'] == 2:
            test_rows.append(row)
        elif row['split'] == 1:
            val_rows.append(row)
        else:
            raise ValueError("Invalid split value", row['split'])

    print("numtrain:", len(train_rows))
    print("numtest:", len(test_rows))
    print("numval:", len(val_rows))

    train_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in train_rows]
    test_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in test_rows]
    val_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in val_rows]

    # save gt as pickle
    with open(cfg.save_dir + 'data/waterbird_imbal95_gt_train.pkl', 'wb') as f:
        pickle.dump(train_img_path_and_gt, f)
    with open(cfg.save_dir + 'data/waterbird_imbal95_gt_test.pkl', 'wb') as f:
        pickle.dump(test_img_path_and_gt, f)
    with open(cfg.save_dir + 'data/waterbird_imbal95_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_gt, f)

    # parse unbalanced metadata
    metadata_csv = pandas.read_csv(cfg.root_dir + 'metadata.csv')
    metadata_csv["category"] = metadata_csv["img_filename"].apply(lambda x: int(x.split(".")[0]))
    metadata_csv['img_filename_no_path'] = metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    train_img_folder = cfg.root_dir + 'train/real/'
    train_metadata_csv = pandas.read_csv(cfg.root_dir + 'train_real_metadata.csv')
    train_metadata_csv['img_filename_no_path'] = train_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    train_test_img_files = []
    gt_train_test = []
    for _, row in train_metadata_csv.iterrows():
        train_test_img_files.append(train_img_folder + row['img_filename_no_path'])
        category = metadata_csv[metadata_csv["img_filename_no_path"] == row['img_filename_no_path']]['category'].values[0]
        gt_train_test.append(category)
    
    test_img_folder = cfg.root_dir + 'test/'
    test_metadata_csv = pandas.read_csv(cfg.root_dir + 'test_metadata.csv')
    test_metadata_csv['img_filename_no_path'] = test_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    for _, row in test_metadata_csv.iterrows():
        train_test_img_files.append(test_img_folder + row['img_filename_no_path'])
        category = metadata_csv[metadata_csv["img_filename_no_path"] == row['img_filename_no_path']]['category'].values[0]
        gt_train_test.append(category)
    
    val_img_folder = cfg.root_dir + 'val/'
    val_metadata_csv = pandas.read_csv(cfg.root_dir + 'val_metadata.csv')
    val_metadata_csv['img_filename_no_path'] = val_metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    val_img_files = []
    gt_val = []
    for _, row in val_metadata_csv.iterrows():
        val_img_files.append(val_img_folder + row['img_filename_no_path'])
        category = metadata_csv[metadata_csv["img_filename_no_path"] == row['img_filename_no_path']]['category'].values[0]
        gt_val.append(category)

    train_test_img_path_and_gt = [(f, gt) for f, gt in zip(train_test_img_files, gt_train_test)]
    val_img_path_and_gt = [(f, gt) for f, gt in zip(val_img_files, gt_val)]
    
    # save gt as pickle
    with open(cfg.save_dir + 'data/waterbird_cat_gt_train_test.pkl', 'wb') as f:
        pickle.dump(train_test_img_path_and_gt, f)
    with open(cfg.save_dir + 'data/waterbird_cat_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_gt, f)

if __name__ == "__main__":
    main()