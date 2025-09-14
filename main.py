import os
import re
import json
import random
import numpy as np
import emoji
from typing import List, Dict, Tuple, Union
from sklearn.decomposition import PCA
from bs4 import BeautifulSoup as bs
import torch
import torch.nn as nn
import pandas as pd


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def list_all_files(folder_path: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


class DataProcessing:
    """Clean the tweet data and create a new folder cleaned_tweet with cleaned data"""

    def __init__(self, input_dir: str = "./tweet", output_dir: str = "./cleaned_tweet"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "positive"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "positive"), exist_ok=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pos_vocab = set()
        self.neg_vocab = set()

    def remain_capital_words(self, content: str) -> str:
        # If the words have more than one capital letter, keep it as is; otherwise, convert to lowercase
        words = content.split()
        return " ".join(
            [
                word if sum(1 for ch in word if ch.isupper()) > 1 else word.lower()
                for word in words
            ]
        )

    def data_cleaning(self, content: str) -> str:

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # remove emoji
        content = emoji.demojize(content)

        content = self.remain_capital_words(content)

        content = re.sub(r"[^\w\s]", "", content)

        # Need the re to remove links like http, https, www
        content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)

        return content.strip()

    def save_cleaned_file(self):
        for subset in ["train", "test"]:
            all_files = list_all_files(os.path.join(self.input_dir, subset))
            for file_path in all_files:
                content = self.data_cleaning(read_file(file_path))
                label = os.path.basename(os.path.dirname(file_path))

                if subset == "train" and content != "":
                    tokens = content.split()
                    if label == "positive":
                        self.pos_vocab.update(tokens)
                    else:
                        self.neg_vocab.update(tokens)

                relpath = os.path.relpath(file_path, self.input_dir)
                output_path = os.path.join(self.output_dir, relpath)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
        pos_word2idx = {word: idx for idx, word in enumerate(self.pos_vocab)}
        neg_word2idx = {word: idx for idx, word in enumerate(self.neg_vocab)}
        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positive": {
                        "vocabulary": list(self.pos_vocab),
                        "word2idx": pos_word2idx,
                        "idx2word": {idx: word for word, idx in pos_word2idx.items()},
                    },
                    "negative": {
                        "vocabulary": list(self.neg_vocab),
                        "word2idx": neg_word2idx,
                        "idx2word": {idx: word for word, idx in neg_word2idx.items()},
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


class PrepareData:
    """Prepare the data for the Neural Network and logistic regression model."""

    def __init__(
        self,
        vocab_path: str = "vocab.json",
    ):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab_dict = json.load(f)

    def generate_neg_samples(
        self, tr_s: List[str], context_window: int, label: str = "positive"
    ) -> Tuple[List[List[str]], List[int]]:
        """Generate negative samples for a positive samples. 2 * context_window negative samples."""
        neg_samples = []

        while len(neg_samples) < 2 * context_window:
            negative_word = random.choice(self.vocab_dict[label]["vocabulary"])

            if (
                negative_word not in tr_s
                and [tr_s[0], negative_word] not in neg_samples
            ):
                neg_samples.append([tr_s[0], negative_word])

        return (neg_samples, [0] * len(neg_samples))

    def create_data(
        self,
        data: List[str],
        context_window: int = 2,
        proximity: bool = False,
        label: str = "positive",
    ) -> Tuple[List[int], List[int]]:
        """Algorithm to generate the training data and labels."""

        tr_s = []
        lab_s = []

        k = 0
        try:
            while k < context_window:
                k += 1
                # Determine the positive label based on the distance with the target word
                if proximity:
                    positive_label = context_window - k + 1
                else:
                    positive_label = 1

                for i in range(len(data)):

                    if i - k < 0 and i + k < len(data) - 1:
                        sample = [data[i], data[i + k]]
                        if sample in tr_s:
                            continue

                        tr_s.append(sample)
                        lab_s.append(positive_label)

                        # generate negative samples (4*[target, negative_word], [0, 0, 0, 0])
                        generate_neg_samples = self.generate_neg_samples(
                            sample, context_window, label
                        )
                        tr_s += generate_neg_samples[0]
                        lab_s += generate_neg_samples[1]

                    elif i + k > len(data) - 1:
                        sample = [data[i], data[i - k]]
                        if sample in tr_s:
                            continue
                        tr_s.append(sample)
                        lab_s.append(positive_label)

                        generate_neg_samples = self.generate_neg_samples(
                            sample, context_window
                        )
                        tr_s += generate_neg_samples[0]
                        lab_s += generate_neg_samples[1]

                    else:
                        r_sample = [data[i], data[i - k]]
                        l_sample = [data[i], data[i + k]]

                        if r_sample not in tr_s:
                            tr_s.append(r_sample)
                            lab_s.append(positive_label)

                            generate_neg_samples = self.generate_neg_samples(
                                r_sample, context_window
                            )
                            tr_s += generate_neg_samples[0]
                            lab_s += generate_neg_samples[1]

                        if l_sample not in tr_s:
                            tr_s.append(l_sample)
                            lab_s.append(positive_label)

                            generate_neg_samples = self.generate_neg_samples(
                                l_sample, context_window
                            )
                            tr_s += generate_neg_samples[0]
                            lab_s += generate_neg_samples[1]

        except Exception as e:
            # For debugging purposes
            print(f"Error in create_data at k={k}, i={i}: {e}")
            print(f"Data: {data}")

        return (tr_s, lab_s)

    def load_data(
        self, label: str = "positive", file_dir: str = "train"
    ) -> Dict[str, Union[List[str], Dict[str, int], Dict[int, str]]]:
        """Load the cleaned data and vocabulary."""

        word2idx = self.vocab_dict[label]["word2idx"]
        idx2word = self.vocab_dict[label]["idx2word"]
        vocab = self.vocab_dict[label]["vocabulary"]
        contents = [
            read_file(content)
            for content in list_all_files(
                os.path.join("./cleaned_tweet", file_dir, label)
            )
        ]
        return {
            "contents": contents,
            "vocab": vocab,
            "word2idx": word2idx,
            "idx2word": idx2word,
        }

    def word_to_index(self, word: str, word2idx: Dict[str, int]) -> int:
        """Convert a word to its corresponding index."""
        # If the word is not found, return the index for <UNK> or -1 if <UNK> is also not found
        return word2idx.get(word, word2idx.get("<UNK>", -1))

    def main(
        self,
        context_window: int = 2,
        proximity: bool = False,
        label: str = "positive",
        file_dir: str = "train",
    ) -> pd.DataFrame:
        """Main function to prepare the data."""

        samples = pd.DataFrame(columns=["data", "label", "encoded_data"])

        variables = self.load_data(label=label, file_dir=file_dir)
        all_tr = []
        all_lab = []

        for content in variables["contents"]:
            tokens = content.split()
            try:
                results = self.create_data(tokens, context_window, proximity, label)

                all_tr += results[0]
                all_lab += results[1]
            except Exception as e:
                print(f"Error processing content: {e}")
                print(f"Content: {content}")
                print(f"Tokens: {tokens}")

        samples["data"] = all_tr
        samples["label"] = all_lab
        samples["encoded_data"] = samples["data"].apply(
            lambda x: [self.word_to_index(word, variables["word2idx"]) for word in x]
        )
        print("Successfully created samples.")

        return samples


class SimpleWord2Vec_LogiR(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(SimpleWord2Vec_LogiR, self).__init__()
        # set TWO embeddings for target and context, respectively
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.linear = nn.Linear(2 * embedding_dim, 1)

    def forward(self, inputs):
        # get indices of target and context from inputs, respectively
        target = inputs[:, 0]
        context = inputs[:, 1]

        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        combined = torch.cat([target_emb, context_emb], dim=1)
        out = self.linear(combined)
        out = torch.sigmoid(out)
        return out


class SimpleWord2Vec_FFNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, node_size=64):
        super(SimpleWord2Vec_FFNN, self).__init__()

        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layer1 = nn.Linear(2 * embedding_dim, node_size)
        self.layer2 = nn.Linear(node_size, node_size)
        self.layer3 = nn.Linear(node_size, 1)

    def forward(self, inputs):
        target = inputs[:, 0]
        context = inputs[:, 1]

        target_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        combined = torch.cat([target_emb, context_emb], dim=1)
        out = self.layer1(combined)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.sigmoid(out)
        return out


def train(model, x_train, y_train, lr=0.01, num_epochs=100):
    """Train the model. Get the Embedding weights after training."""
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total_loss = 0
    losses = []
    for epoch in range(num_epochs):
        y_predicted = model(x_train)

        optimizer.zero_grad()
        loss = loss_function(y_predicted, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training complete.\n")
    print("total train_loss =", total_loss)

    # --- Extract embeddings ---
    target_emb = model.target_embedding.weight.detach().numpy()
    context_emb = model.context_embedding.weight.detach().numpy()

    emb_sum = target_emb + context_emb
    emb_avg = (target_emb + context_emb) / 2
    emb_concat = np.concatenate([target_emb, context_emb], axis=1)

    return {"sum": emb_sum, "avg": emb_avg, "concat": emb_concat}, losses


if __name__ == "__main__":
    if os.path.exists("./cleaned_tweet") and os.path.exists("./vocab.json"):
        print(
            "cleaned_tweet folder already exists. Please remove it first if you want to re-generate."
        )
    else:
        processor = DataProcessing()
        processor.save_cleaned_file()
        print("Successfully cleaned the tweet data and saved in cleaned_tweet folder")

    dataloader = PrepareData(vocab_path="vocab.json")

    training_data = dataloader.main(
        context_window=2, proximity=False, label="positive", file_dir="train"
    )
    testing_data = dataloader.main(
        context_window=2, proximity=False, label="positive", file_dir="test"
    )

    vocab = dataloader.load_data(label="positive", file_dir="train")

    x_train = torch.tensor(training_data["encoded_data"].tolist(), dtype=torch.long)
    y_train = torch.tensor(
        training_data["label"].tolist(), dtype=torch.float32
    ).unsqueeze(1)

    log_model = SimpleWord2Vec_LogiR(vocab_size=len(vocab["vocab"]), embedding_dim=64)
    nn_model = SimpleWord2Vec_FFNN(
        vocab_size=len(vocab["vocab"]), embedding_dim=64, node_size=32
    )

    log_embeddings, log_losses = train(log_model, x_train, y_train)
    nn_embeddings, nn_losses = train(nn_model, x_train, y_train)

    breakpoint()

    # On going...
    # Evaluate the model on the test set
    # Extract the weights and plot on 2D space using PCA
