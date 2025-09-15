import os
import re
import json
import random
import numpy as np
import emoji
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
from sklearn.manifold import TSNE
from bs4 import BeautifulSoup as bs
import torch
import torch.nn as nn
import pandas as pd
import logging

words_to_plot = [
    "hire",
    "ATC",
    "worst2unitedflightsever",
    "cha",
    "lusaka",
    "share",
    "excuse",
    "badpolicy",
    "exception",
    "annricord",
    "GRU",
    "HOPELESS",
    "1735",
    "reopen",
    "unitedfailsworsttripofmylife",
    "evaluate",
    "thickens",
    "overheating",
    "passport",
    "chosen",
    "yet",
    "crying_face",
    "DENPHX",
    "tuesday",
    "blizzard",
    "rage",
    "frustrating",
    "preparedhonestly",
    "evennotified",
    "quoting",
    "hassle",
    "aware",
    "dpted",
    "workers",
    "giving",
    "customers",
    "PRERECORDED",
    "changing",
    "seems",
    "comeonpeople",
    "5612",
    "face",
    "street",
    "credit",
    "midnight",
    "hoagy10",
    "set",
    "bitcoin",
    "715",
    "55",
    "jana",
    "assigned",
    "thur",
    "scenario",
    "btw",
    "cannot",
    "2plains",
    "exist",
    "backed",
    "joyadventuremom",
    "KPHL",
    "bother",
    "after2",
    "unhappycustomer",
    "resolved",
    "rdu",
    "massages",
    "difference",
    "unhappy",
    "221",
    "mgmt",
    "early",
    "till",
    "filthy",
    "one",
    "brought",
    "stat",
    "raving",
    "report",
    "219",
    "hit",
    "swallowed",
    "645",
    "unprofessionally",
    "want",
    "ser",
    "worse",
    "MC",
    "really",
    "crying",
    "case",
    "inactivity",
    "710",
    "gettin",
    "record",
    "scheduling",
    "any",
    "2a",
    "discuss",
    "bdl",
    "prompts",
]

# Remove log of previous training
os.remove("main.log")

# Configure basic logging to a file
logging.basicConfig(
    filename='main.log',  # Name of the log file
    level=logging.INFO,  # Minimum logging level to capture (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s'  # Format of the log messages
)
logger = logging.getLogger(__name__)

# Simple function to read in files as string
def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        logger.debug(f"read_file: {file_path}")
        return f.read()

# Simple function to list all files under a directory and it's sub-directories as a list of strings
def list_all_files(folder_path: str) -> List[str]:
    files = []
    logger.debug(f"list_all_files: {folder_path}")
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


class DataProcessing:
    """Clean the tweet data and create a new folder cleaned_tweet with cleaned data"""

    """
    The init function of this class sets up the folder structure for cleaned versions of
    the tweets dataset that mirrors it's folder structure. It also performs some sanity checks
    to see if the dataset exists or not
    """
    def __init__(self, input_dir: str = "./tweet", output_dir: str = "./cleaned_tweet"):
        logger = logging.getLogger(__name__)
        if (os.path.isdir(input_dir) and
            os.path.isdir(os.path.join(output_dir, "test", "negative")) and
            os.path.isdir(os.path.join(output_dir, "test", "positive")) and
            os.path.isdir(os.path.join(output_dir, "train", "negative")) and
            os.path.isdir(os.path.join(output_dir, "train", "positive"))):
            raise Exception("The tweet folder doesn't exist or is corrupted. Please check the folder and try again.")

        logger.info("Creating cleaned_tweet folder structure")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", "positive"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "negative"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train", "positive"), exist_ok=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pos_vocab = set()
        self.neg_vocab = set()
        self.pos_vocab.add("<UNK>")
        self.neg_vocab.add("<UNK>")

    # If the words have more than one capital letter, keep it as is; otherwise, convert to lowercase
    def remain_capital_words(self, content: str) -> str:
        words = content.split()
        return " ".join(
            [
                word if sum(1 for ch in word if ch.isupper()) > 1 else word.lower()
                for word in words
            ]
        )

    # Main data cleaning function
    def data_cleaning(self, content: str) -> str:

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # remove emoji
        content = emoji.demojize(content)

        # Removes capitalization
        content = self.remain_capital_words(content)

        # Removes punctuations
        content = re.sub(r"[^\w\s]", "", content)

        # Need the re to remove links like http, https, www
        content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)

        return content.strip()

    """
    This function first performs data cleaning, then save all cleaned tweets into the cleaned_tweet folder,
    mirroring the tweet folder structure. Then, it creates word-to-id and id-to-word mappings and saves everything
    into vocab.json.
    """
    def save_cleaned_file(self):
        for subset in ["train", "test"]:
            logger.info(f"Cleaning data on {subset} dataset")
            all_files = list_all_files(os.path.join(self.input_dir, subset))
            for file_path in all_files:
                # Get the content of the sample and label from folder name
                content = self.data_cleaning(read_file(file_path))
                label = os.path.basename(os.path.dirname(file_path))

                # Tokenized clean data and put them into appropriate folder from label
                if subset == "train" and content != "":
                    tokens = content.split()
                    if label == "positive":
                        self.pos_vocab.update(tokens)
                    else:
                        self.neg_vocab.update(tokens)

                # Getting the relative path to input directory and mirroring it to output directory
                relpath = os.path.relpath(file_path, self.input_dir)
                output_path = os.path.join(self.output_dir, relpath)

                # Write cleaned tweet into cleaned_tweet according to label
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

        # Create id mapping for positive and negative words for entire vocabulary
        pos_word2idx = {word: idx for idx, word in enumerate(self.pos_vocab)}
        neg_word2idx = {word: idx for idx, word in enumerate(self.neg_vocab)}
        logger.info("Created word-to-id mapping for entire vocabulary")

        # Write the entire vocabulary including the word-to-id and id-to-word mappings to vocab.json
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
        logger.info("Finished writting to vocab.json")


class PrepareData:
    """Prepare the data for the Neural Network and logistic regression model."""

    # The init function simply prepares the vocab dict by reading in vocab.json prepared by the DataProcessing class
    def __init__(
        self,
        vocab_path: str = "vocab.json",
    ):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab_dict = json.load(f)

    """
    This class generates negative samples by randomly selecting words in the vocabulary not including
    the target word and its context
    """
    def generate_neg_samples(
        self,
        target: str,
        exclusion: List[str],
        context_window: int,
        label: str = "positive"
    ) -> Tuple[List[List[str]], List[int]]:
        """Generate negative samples for a positive samples. 2 * context_window negative samples."""
        neg_samples = []

        while len(neg_samples) < 2 * context_window:
            negative_word = random.choice(self.vocab_dict[label]["vocabulary"])

            if (
                negative_word != target and
                negative_word not in exclusion
            ):
                neg_samples.append([target, negative_word])

        return (neg_samples, [0] * len(neg_samples))

    """
    Generate positive context pairs by using a sliding window over each target word in a tweet.
    After each target word is feeded, a 2*N random negative samples are created.
    """
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
        try:
            for i in range(len(data)):
                # Establish target word and context words using a sliding window of context_size
                target = data[i]
                positive_label = 1
                context = data[0 if i < context_window else i - context_window : i + context_window + 1]
                context.remove(target) # Remove target word from context
                
                # Add each context pair
                for ctx_word in context:
                    tr_s.append([target, ctx_word])
                    lab_s.append(positive_label)

                # Generate N*2 negative samples for target word
                generate_neg_samples = self.generate_neg_samples(target, context, context_window, label)
                tr_s += generate_neg_samples[0]
                lab_s += generate_neg_samples[1]

        except Exception as e:
            # For debugging purposes
            logger.error(f"Error in create_data {e}")
            logger.error(f"Data: {data}")
            logger.error(f"tr_s: {tr_s}")
            logger.error(f"lab_s: {lab_s}")

        return (tr_s, lab_s)

    # Function to aggregate various generated data to a single data structure
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

    """Convert a word to its corresponding index."""
    def word_to_index(self, word: str, word2idx: Dict[str, int]) -> int:
        # If the word is not found, return the index for <UNK> or -1 if <UNK> is also not found
        return word2idx.get(word, word2idx.get("<UNK>", 0))

    """Main function to prepare the data."""
    def main(
        self,
        context_window: int = 2,
        proximity: bool = False,
        label: str = "positive",
        file_dir: str = "train",
    ) -> pd.DataFrame:

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
                logger.error(f"Error processing content: {e}")
                logger.error(f"Content: {content}")
                logger.error(f"Tokens: {tokens}")

        samples["data"] = all_tr
        samples["label"] = all_lab
        samples["encoded_data"] = samples["data"].apply(
            lambda x: [self.word_to_index(word, variables["word2idx"]) for word in x]
        )
        logger.info("Successfully created samples.")

        return samples

class SimpleWord2Vec_LogiR(nn.Module):
    """
    A simple Word2Vec-style model that uses Logistic Regression on top of
    concatenated word embeddings to predict if a word pair is a positive
    (real context) or negative (random) sample.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        """
        Initializes the model layers.
        Args:
            vocab_size (int): The total number of unique words in the vocabulary.
            embedding_dim (int): The desired dimensionality of the word vectors.
        """
        super(SimpleWord2Vec_LogiR, self).__init__()
        # set TWO embeddings for target and context, respectively
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.linear = nn.Linear(2 * embedding_dim, 1)

    def forward(self, inputs):
        """
        Defines the forward pass of the model.
        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, 2), where
                                   inputs[:, 0] are target word indices and
                                   inputs[:, 1] are context word indices.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) containing the
                          predicted probabilities.
        """
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
    """
    A more complex Word2Vec-style model that uses a small Feed-Forward
    Neural Network (FFNN) instead of a simple logistic regression layer.
    """

    def __init__(self, vocab_size, embedding_dim, node_size=64):
        """
        Initializes the model layers.
        Args:
            vocab_size (int): The total number of unique words in the vocabulary.
            embedding_dim (int): The desired dimensionality of the word vectors.
            node_size (int): The number of neurons in the hidden layers.
        """
        super(SimpleWord2Vec_FFNN, self).__init__()

        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.layer1 = nn.Linear(2 * embedding_dim, node_size)
        self.layer2 = nn.Linear(node_size, node_size)
        self.layer3 = nn.Linear(node_size, 1)

    def forward(self, inputs):
        """
        Defines the forward pass of the model.
        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, 2).
        Returns:
            torch.Tensor: A tensor of shape (batch_size, 1) of probabilities.
        """
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

# Function to train either logistic regression or neural network model
def train(model, x_train, y_train, lr=0.01, num_epochs=100):
    """
    Trains a given PyTorch model and extracts the learned embeddings.
    Args:
        model (nn.Module): The model to be trained (either LogiR or FFNN).
        x_train (torch.Tensor): The input training data (word pair indices).
        y_train (torch.Tensor): The training labels (0 or 1).
        lr (float): The learning rate for the optimizer.
        num_epochs (int): The number of times to iterate over the entire dataset.
    Returns:
        tuple: A tuple containing:
               - A dictionary of the final combined word embeddings.
               - A list of loss values for each epoch.
    """
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total_loss = 0
    losses = []
    logger.info(f"Begin training for\n{model}")
    for epoch in range(num_epochs):
        y_predicted = model(x_train)

        optimizer.zero_grad()
        loss = loss_function(y_predicted, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    logger.info("Training complete.")
    logger.info(f"Total train_loss = {total_loss}\n\n")

    # --- Extract embeddings ---
    target_emb = model.target_embedding.weight.detach().numpy()
    context_emb = model.context_embedding.weight.detach().numpy()

    emb_sum = target_emb + context_emb
    emb_avg = (target_emb + context_emb) / 2
    emb_concat = np.concatenate([target_emb, context_emb], axis=1)

    return {"sum": emb_sum, "avg": emb_avg, "concat": emb_concat}, losses

# Function to evaluate model based on loss and accuracy
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)

        loss_fn = nn.BCELoss()
        loss = loss_fn(y_pred, y_test).item()

        preds = (y_pred >= 0.5).float()
        accuracy = (preds == y_test).float().mean().item()

    return loss, accuracy

# Function to plot embeddings based on selected vocabulary
def plot_embeddings(
    method: str = "avg", initial_embeddings=None, train_embeddings=None, vocab=None
):
    """Plot only the embeddings for the selected words using t-SNE."""

    word2idx = vocab["word2idx"]
    indices = [word2idx.get(word, word2idx["<UNK>"]) for word in words_to_plot]

    # Select only the embeddings for the words to plot
    init_selected = initial_embeddings[indices]
    trained_selected = train_embeddings[method][indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    init_emb_2d = tsne.fit_transform(init_selected)
    trained_emb_2d = tsne.fit_transform(trained_selected)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(init_emb_2d[:, 0], init_emb_2d[:, 1], alpha=0.6)
    for i, word in enumerate(words_to_plot):
        plt.annotate(word, (init_emb_2d[i, 0], init_emb_2d[i, 1]), color="red")
    plt.title("Initial Embeddings (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.subplot(1, 2, 2)
    plt.scatter(trained_emb_2d[:, 0], trained_emb_2d[:, 1], alpha=0.6)
    for i, word in enumerate(words_to_plot):
        plt.annotate(word, (trained_emb_2d[i, 0], trained_emb_2d[i, 1]), color="red")
    plt.title("Trained Embeddings (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.tight_layout()
    plt.title("Embeddings plot")
    plt.savefig("embeddings.png")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if os.path.exists("./cleaned_tweet") and os.path.exists("./vocab.json"):
        logger.warning(
            "cleaned_tweet folder already exists. Please remove it first if you want to re-generate."
        )
    else:
        processor = DataProcessing()
        processor.save_cleaned_file()
        logger.info("Successfully cleaned the tweet data and saved in cleaned_tweet folder")

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

    x_test = torch.tensor(testing_data["encoded_data"].tolist(), dtype=torch.long)
    y_test = torch.tensor(
        testing_data["label"].tolist(), dtype=torch.float32
    ).unsqueeze(1)

    log_model = SimpleWord2Vec_LogiR(vocab_size=len(vocab["vocab"]), embedding_dim=64)
    nn_model = SimpleWord2Vec_FFNN(
        vocab_size=len(vocab["vocab"]), embedding_dim=64, node_size=32
    )

    log_init_target = log_model.target_embedding.weight.detach().numpy()
    log_init_context = log_model.context_embedding.weight.detach().numpy()
    log_init_avg = (log_init_target + log_init_context) / 2

    log_embeddings, log_losses = train(log_model, x_train, y_train)
    nn_embeddings, nn_losses = train(nn_model, x_train, y_train)

    # Logistic Regression model
    log_test_loss, log_test_acc = evaluate(log_model, x_test, y_test)
    logger.info(f"LogiR Test Loss: {log_test_loss:.4f}, Accuracy: {log_test_acc:.4f}")

    # Feedforward NN model
    nn_test_loss, nn_test_acc = evaluate(nn_model, x_test, y_test)
    logger.info(f"FFNN Test Loss: {nn_test_loss:.4f}, Accuracy: {nn_test_acc:.4f}")

    plot_embeddings(
        method="avg",
        initial_embeddings=log_init_avg,
        train_embeddings=log_embeddings,
        vocab=vocab,
    )
