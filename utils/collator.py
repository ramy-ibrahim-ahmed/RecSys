import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def collate_fn(batch: list[dict]):
    batch_size = len(batch)

    userid = torch.stack([b["userid"] for b in batch])
    movieid = torch.stack([b["movieid"] for b in batch])
    dense_features = torch.stack([b["gender"] for b in batch]).unsqueeze(1)
    labels = torch.stack([b["label"] for b in batch])

    genre_tensors = [b["genres"] for b in batch]
    genre_values = torch.cat(genre_tensors)
    genre_lengths = [b["genres_len"] for b in batch]

    sparse_features = KeyedJaggedTensor(
        keys=["userid", "movieid", "genres"],
        values=torch.cat([userid, movieid, genre_values]),
        lengths=torch.tensor(
            [1] * batch_size + [1] * batch_size + genre_lengths, dtype=torch.int32
        ),
    )

    return sparse_features, dense_features, labels
