import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = row.user_id
        sex = row.sex
        age = row.age_group
        occupation = row.occupation
        input_seq = row.input_seq
        target_item = row.target_item
        label = row.is_clicked

        user_feat = (user_id, sex, age, occupation)
        seq_item = torch.tensor(input_seq)

        return (user_feat, seq_item, target_item, label)
