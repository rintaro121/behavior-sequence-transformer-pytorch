import torch


class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 5e-4
    n_epoch = 10

    num_user = 6040
    num_movie = 3883
    num_sex = 2
    num_age_group = 7
    num_occupation = 21

    seq_len = 4
    num_layers = 2

    user_emb_dim = 32
    movie_emb_dim = 32
    sex_emb_dim = 2
    age_group_emb_dim = 4
    occupation_emb_dim = 4

    d_model = 32
    nhead = 2
    dim_feedforward = 64

    mlp_dim = (
        d_model
        + user_emb_dim
        + sex_emb_dim
        + age_group_emb_dim
        + occupation_emb_dim
        + movie_emb_dim
    )
    mlp_hidden_1_dim = 128
    mlp_hidden_2_dim = 64
    mlp_hidden_3_dim = 32
    mlp_hidden_4_dim = 1

    dropout_rate = 0.1
