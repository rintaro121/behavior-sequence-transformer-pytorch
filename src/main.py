import os
import random

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import MovieLensDataset
from model import LightningModule

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
BASE_DIR = "./dataset"
DATASET_DIR = os.path.join(BASE_DIR, "ml-1m")
OUTPUT_PATH = os.path.join(BASE_DIR, "sequential_dataset.csv")


def add_target_item(x, all_movies_set):
    # Add negative items for CTR prediction
    movie_seq_data = x["movie_seq"]
    input_seq_len = len(movie_seq_data) - 1
    input_seq = movie_seq_data[:input_seq_len]

    if np.random.random() <= 0.5:
        is_clicked = True
        target_item = movie_seq_data[-1]
    else:
        is_clicked = False
        negative_items = list(all_movies_set - set(movie_seq_data))
        target_item = random.choice(negative_items)
    return input_seq, is_clicked, target_item


def main():
    print("===== 1. Preprocess =====")

    user_filepath = os.path.join(DATASET_DIR, "users.dat")
    movie_filepath = os.path.join(DATASET_DIR, "movies.dat")
    rating_filepath = os.path.join(DATASET_DIR, "ratings.dat")

    users = pd.read_csv(
        user_filepath,
        sep="::",
        names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        dtype={
            "user_id": int,
            "sex": str,
            "age_group": int,
            "occupation": int,
            "zip_code": str,
        },
    )

    movies = pd.read_csv(
        movie_filepath,
        sep="::",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": int, "title": str, "genres": str},
        encoding="ISO-8859-1",
    )
    ratings = pd.read_csv(
        rating_filepath,
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"],
        dtype={"user_id": int, "movie_id": int, "rating": int, "unix_timestamp": int},
    )

    # user feat
    unique_user_ids = users["user_id"].unique()
    unique_user_ids.sort()

    unique_sex = users["sex"].unique()
    unique_sex.sort()

    unique_age_group = users["age_group"].unique()
    unique_age_group.sort()

    unique_ocuupation = users["occupation"].unique()
    unique_ocuupation.sort()

    # movie feat
    unique_movie_ids = movies["movie_id"].unique()
    unique_movie_ids.sort()

    # defaine mapping dict
    user_id_mapping = {user_id: i for i, user_id in enumerate(unique_user_ids)}
    sex_mapping = {sex: i for i, sex in enumerate(unique_sex)}
    age_group_mapping = {age_group: i for i, age_group in enumerate(unique_age_group)}
    occupation_mapping = {
        occupation: i for i, occupation in enumerate(unique_ocuupation)
    }
    movie_id_mapping = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}

    ratings["user_id"] = ratings["user_id"].map(user_id_mapping)
    ratings["movie_id"] = ratings["movie_id"].map(movie_id_mapping)

    movies["movie_id"] = movies["movie_id"].map(movie_id_mapping)

    users["user_id"] = users["user_id"].map(user_id_mapping)
    users["sex"] = users["sex"].map(sex_mapping)
    users["age_group"] = users["age_group"].map(age_group_mapping)
    users["occupation"] = users["occupation"].map(occupation_mapping)

    ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

    ratings_data = pd.DataFrame(
        data={
            "user_id": list(ratings_group.groups.keys()),
            "movie_ids": list(ratings_group.movie_id.apply(list)),
            "ratings": list(ratings_group.rating.apply(list)),
            "timestamps": list(ratings_group.unix_timestamp.apply(list)),
        }
    )

    # Generate sequence data
    user_id_data = []
    movie_seq_data = []
    rating_seq_data = []

    sequence_length = 5
    window_size = 1

    print("Generating user sequences")
    for i in tqdm(range(len(ratings_data))):
        row = ratings_data.iloc[i]

        movie_id_history = torch.tensor(row.movie_ids)
        rating_history = torch.tensor(row.ratings)
        movie_ids_seq = (
            movie_id_history.ravel()
            .unfold(0, sequence_length, window_size)
            .to(torch.int32)
        )
        ratings_seq = (
            rating_history.ravel()
            .unfold(0, sequence_length, window_size)
            .to(torch.int32)
        )

        user_id_data += [row.user_id] * len(movie_ids_seq)
        movie_seq_data += movie_ids_seq.tolist()
        rating_seq_data += ratings_seq.tolist()

    sequencd_data_df = pd.DataFrame(
        {
            "user_id": user_id_data,
            "movie_seq": movie_seq_data,
            "rating_seq": rating_seq_data,
        }
    )

    all_movies_set = set(movies.movie_id)
    print("Add negative items")
    sequencd_data_df[["input_seq", "is_clicked", "target_item"]] = (
        sequencd_data_df.apply(
            lambda x: add_target_item(x, all_movies_set), axis=1, result_type="expand"
        )
    )
    sequencd_data_df = sequencd_data_df.join(users.set_index("user_id"), on="user_id")

    print("===== 2.Make dataset for BST =====")
    random_selection = np.random.rand(len(sequencd_data_df)) <= 0.85
    train_df = sequencd_data_df[random_selection]
    valid_df = sequencd_data_df[~random_selection]

    print(f"Train Size : {len(train_df)}")
    print(f"Valid Size : {len(valid_df)}")

    train_dataset = MovieLensDataset(train_df)
    valid_dataset = MovieLensDataset(valid_df)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
    )

    mlf_logger = MLFlowLogger(
        experiment_name="BehaviorSequenceTransomer_logs",
        run_name="exp001",
        log_model="all",
    )
    model = LightningModule(CFG=CFG)
    trainer = L.Trainer(
        max_epochs=CFG.n_epoch,
        logger=mlf_logger,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
