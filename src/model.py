import math

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim


class BST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(cfg)
        self.transformer_layer = TransformerLayer(cfg)
        self.mlp_layer = MLPLayer(cfg)

    def forward(self, user_feat, seq_item, target_item):
        user_emb, seq_item_emb, target_item_emb = self.embedding_layer(
            user_feat, seq_item, target_item
        )

        transformer_output = self.transformer_layer(seq_item_emb)

        concat_feat = torch.concat(
            [user_emb, transformer_output, target_item_emb],
            dim=-1,
        )

        p_ctr = self.mlp_layer(concat_feat)
        return p_ctr


class EmbeddingLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.user_embedding = nn.Embedding(cfg.num_user, cfg.user_emb_dim)
        self.sex_embedding = nn.Embedding(cfg.num_sex, cfg.sex_emb_dim)
        self.age_embedding = nn.Embedding(cfg.num_age_group, cfg.age_group_emb_dim)
        self.occupation_embedding = nn.Embedding(
            cfg.num_occupation, cfg.occupation_emb_dim
        )
        self.movie_embedding = nn.Embedding(cfg.num_movie, cfg.movie_emb_dim)

    def forward(
        self,
        user_feat,
        seq_item,
        target_item,
    ):
        # Get user embeddings
        user_id, sex, age, occupation = user_feat
        user_emb = self.user_embedding(user_id)
        sex_emb = self.sex_embedding(sex)
        age_emb = self.age_embedding(age)
        occupation_emb = self.occupation_embedding(occupation)

        # Get movie embedding
        seq_item_emb = self.movie_embedding(seq_item)
        target_item_emb = self.movie_embedding(target_item)

        user_feat = torch.concat([user_emb, sex_emb, age_emb, occupation_emb], dim=-1)

        return user_feat, seq_item_emb, target_item_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x):
        """
        x:Tensor, shape(batch size, seq_len, emb_dim)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pe = PositionalEncoding(
            d_model=cfg.d_model,
            dropout=cfg.dropout_rate,
            max_len=cfg.seq_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )

    def forward(self, movie_seq_emb):
        x = self.pe(movie_seq_emb)
        enc_out = self.transformer_encoder(x)[:, -1, :]
        return enc_out


class MLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.mlp_dim, cfg.mlp_hidden_1_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_1_dim, cfg.mlp_hidden_2_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_2_dim, cfg.mlp_hidden_3_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.mlp_hidden_3_dim, cfg.mlp_hidden_4_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.mlp(x)
        return output


class LightningModule(L.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.model = BST(CFG)
        self.loss_fn = nn.BCELoss()
        self.validation_step_outputs = []
        self.validation_step_labels = []

    def forward(self, user_feat, input_seq, target_item):
        return self.model(user_feat, input_seq, target_item)

    def training_step(self, batch, batch_idx):
        user_feat, input_seq, target_item, label = batch

        output = self(user_feat, input_seq, target_item)
        output = output.squeeze(-1)
        label = label.type(torch.float32)

        loss = self.loss_fn(output, label)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        user_feat, input_seq, target_item, label = batch

        output = self(user_feat, input_seq, target_item)
        output = output.squeeze(-1)
        label = label.type(torch.float32)

        self.validation_step_outputs.append(output)
        self.validation_step_labels.append(label)

        loss = self.loss_fn(output, label)
        self.log("valid_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        all_output = torch.concat(self.validation_step_outputs)
        all_label = torch.concat(self.validation_step_labels)
        hit_count = ((all_output > 0.5) == all_label).sum()
        accuracy = hit_count / len(all_output)
        self.log("accuracy", accuracy)

        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.CFG.lr)
        return optimizer
