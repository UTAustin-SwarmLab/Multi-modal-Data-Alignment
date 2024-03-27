import pickle

import pandas
from omegaconf import DictConfig

import hydra


@hydra.main(version_base=None, config_path='config', config_name='main_config')
def main(cfg: DictConfig):
    metadata_csv = pandas.read_csv(cfg.root95_dir + 'metadata.csv')
    metadata_csv['img_filename_no_path'] = metadata_csv['img_filename'].apply(lambda x: x.split('/')[-1])
    metadata_csv["category"] = metadata_csv["img_filename"].apply(lambda x: int(x.split(".")[0]))
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

    # binary ground truth 
    train_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in train_rows]
    test_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in test_rows]
    val_img_path_and_gt = [(row["img_filename_no_path"], row["y"]) for row in val_rows]

    # save gt as pickle
    with open(cfg.save_path + 'data/waterbird_imbal95_gt_train.pkl', 'wb') as f:
        pickle.dump(train_img_path_and_gt, f)
    with open(cfg.save_path + 'data/waterbird_imbal95_gt_test.pkl', 'wb') as f:
        pickle.dump(test_img_path_and_gt, f)
    with open(cfg.save_path + 'data/waterbird_imbal95_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_gt, f)

    # category ground truth
    train_img_path_and_cat_bg = [(row["img_filename_no_path"], row["category"]) for row in train_rows]
    test_img_path_and_cat_bg = [(row["img_filename_no_path"], row["category"]) for row in test_rows]
    val_img_path_and_cat_bg = [(row["img_filename_no_path"], row["category"]) for row in val_rows]

    # save gt as pickle
    with open(cfg.save_path + 'data/waterbird_imbal95_cat_gt_train.pkl', 'wb') as f:
        pickle.dump(train_img_path_and_cat_bg, f)
    with open(cfg.save_path + 'data/waterbird_imbal95_cat_gt_test.pkl', 'wb') as f:
        pickle.dump(test_img_path_and_cat_bg, f)
    with open(cfg.save_path + 'data/waterbird_imbal95_cat_gt_val.pkl', 'wb') as f:
        pickle.dump(val_img_path_and_cat_bg, f)    

if __name__ == "__main__":
    main()