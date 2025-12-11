# __Data_Set__
# The tensor objects for GPU processing
# Return: a dict of tensors
import torch
import pandas as pd
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.userid = torch.tensor(dataframe["userid"].values, dtype=torch.long)
        self.movieid = torch.tensor(dataframe["movieid"].values, dtype=torch.long)
        self.gender = torch.tensor(dataframe["gender"].values, dtype=torch.float32)
        self.love = torch.tensor(dataframe["rating_binary"].values, dtype=torch.float32)

        self.genre_lists = [
            torch.tensor(g, dtype=torch.long) for g in dataframe["genre_list"].values
        ]

    def __len__(self):
        return len(self.userid)

    def __getitem__(self, index):
        return {
            "userid": self.userid[index],
            "movieid": self.movieid[index],
            "gender": self.gender[index],
            "genres": self.genre_lists[index],
            "genres_len": len(self.genre_lists[index]),
            "label": self.love[index],
        }
