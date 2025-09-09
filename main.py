import os
import re
import json
import random
from typing import List, Dict, Tuple
import emoji
from bs4 import BeautifulSoup as bs
import pandas as pd


class DataProcessing:
    def __init__(
        self, train_dir: str = "./tweet/train", test_dir: str = "./tweet/test"
    ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.training = pd.DataFrame(columns=["content", "label"])
        self.testing = pd.DataFrame(columns=["content", "label"])

    def list_all_files(self, folder_path: str) -> List[str]:
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def read_file(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def remain_capital_words(self, content: str) -> str:
        # If the words have more than one capital letter, keep it as is; otherwise, convert to lowercase
        words = content.split()
        return " ".join(
            [
                word if sum(1 for ch in word if ch.isupper()) > 1 else word.lower()
                for word in words
            ]
        )

    def tokenize_and_remove_punc(self, content: str) -> List[str]:
        # Remove punctuation (keep only letters, numbers, and whitespace)
        content = re.sub(r"[^\w\s]", "", content)

        tokens = content.split()
        return tokens

    def data_cleaning(self, content: str) -> str:

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # remove emoji
        content = emoji.demojize(content)

        content = self.remain_capital_words(content)

        content = re.sub(r"[^\w\s]", "", content)

        return content.strip()

    def build_vocabulary(self, label: int) -> set:
        """
        Build vocabulary from texts with specified label. And store the pandas DataFrame as csv file.
        """

        filtered_data = self.training[self.training["label"] == label].copy()

        # Store the tokens into column just to check the result in csv file
        filtered_data["tokens"] = filtered_data["content"].apply(
            self.tokenize_and_remove_punc
        )

        filename = (
            "positive_label_data.csv" if label == 1 else "negative_label_data.csv"
        )

        filtered_data.to_csv(filename, index=False)
        print(f"Successfully saved {filename}")

        vocab = set()
        for tokens in filtered_data["tokens"]:
            vocab.update(tokens)

        return vocab

    def build_dataframe(self, file_dir: str) -> pd.DataFrame:
        """Build a pandas DataFrame from files in the specified directory."""
        file_paths = self.list_all_files(file_dir)
        rows = []
        for path in file_paths:
            rows.append(
                {
                    "content": self.read_file(path),
                    "label": (
                        1
                        if os.path.basename(os.path.dirname(path)) == "positive"
                        else 0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def main(self) -> Dict[str, any]:

        self.training = self.build_dataframe(self.train_dir)

        self.training["content"] = self.training["content"].apply(self.data_cleaning)

        positive_vocab = self.build_vocabulary(label=1)
        word2idx = {word: idx for idx, word in enumerate(positive_vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}

        result = {
            "positive_vocab": list(positive_vocab),
            "word2idx": word2idx,
            "idx2word": idx2word,
        }

        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("Successfully saved vocab.json")
        return result


class PrepareTrainingData:
    def __init__(
        self, vocab: Dict[str, any], file_dir: str = "positive_label_data.csv"
    ):
        self.word2idx = vocab["word2idx"]
        self.idx2word = vocab["idx2word"]
        self.vocab = vocab["positive_vocab"]
        self.training_data = pd.read_csv(file_dir)
        self.positive_samples = pd.DataFrame(columns=["training", "label"])

    def generate_neg_samples(
        self, tr_s: List[str], context_window: int
    ) -> Tuple[List[List[str]], List[int]]:
        """Generate negative samples for a positive samples. 2 * context_window negative samples."""
        neg_samples = []

        while len(neg_samples) < 2 * context_window:
            negative_word = random.choice(self.vocab)

            if (
                negative_word not in tr_s
                and [tr_s[0], negative_word] not in neg_samples
            ):
                neg_samples.append([tr_s[0], negative_word])

        return (neg_samples, [0] * len(neg_samples))

    def create_training_data(
        self, data: List[str], context_window: int = 2
    ) -> Tuple[List[int], List[int]]:
        """Algorithm to generate the training data and labels."""

        tr_s = []
        lab_s = []

        k = 0
        try:
            while k < context_window:
                k += 1
                for i in range(len(data)):

                    if i - k < 0 and i + k < len(data) - 1:
                        sample = [data[i], data[i + k]]
                        if sample in tr_s:
                            continue

                        tr_s.append(sample)
                        lab_s.append(context_window - k + 1)

                        # generate negative samples (4*[target, negative_word], [0, 0, 0, 0])
                        generate_neg_samples = self.generate_neg_samples(
                            sample, context_window
                        )
                        tr_s += generate_neg_samples[0]
                        lab_s += generate_neg_samples[1]

                    elif i + k > len(data) - 1:
                        sample = [data[i], data[i - k]]
                        if sample in tr_s:
                            continue
                        tr_s.append(sample)
                        lab_s.append(context_window - k + 1)

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
                            lab_s.append(context_window - k + 1)

                            generate_neg_samples = self.generate_neg_samples(
                                r_sample, context_window
                            )
                            tr_s += generate_neg_samples[0]
                            lab_s += generate_neg_samples[1]

                        if l_sample not in tr_s:
                            tr_s.append(l_sample)
                            lab_s.append(context_window - k + 1)

                            generate_neg_samples = self.generate_neg_samples(
                                l_sample, context_window
                            )
                            tr_s += generate_neg_samples[0]
                            lab_s += generate_neg_samples[1]

        except Exception as e:
            # For debugging purposes
            print(f"Error in create_training_data at k={k}, i={i}: {e}")
            print(f"Data: {data}")

        return (tr_s, lab_s)

    def main(self, context_window: int = 2) -> pd.DataFrame:
        all_tr = []
        all_lab = []

        for _, row in self.training_data.iterrows():
            tokens = row["content"].split(" ")
            try:
                results = self.create_training_data(tokens, context_window)

                all_tr += results[0]
                all_lab += results[1]
            except Exception as e:
                print(f"Error processing row {_}: {e}")
                print(f"Content: {row['content']}")
                print(f"Tokens: {tokens}")

        self.positive_samples["training"] = all_tr
        self.positive_samples["label"] = all_lab
        print("Successfully created training samples.")

        return self.positive_samples


if __name__ == "__main__":
    # If exist skip the data processing step
    if not os.path.exists("positive_label_data.csv") or not os.path.exists(
        "vocab.json"
    ):
        processor = DataProcessing()
        processor.main()
    else:
        print(
            "positive_label_data.csv and vocab.json already exists. Skipping data processing."
        )
        positive_data = pd.read_csv("positive_label_data.csv")
        vocab = json.load(open("vocab.json", "r", encoding="utf-8"))

    trainer = PrepareTrainingData(vocab)
    df = trainer.main(context_window=2)

    print("Total rows:", df.shape[0])
    print(df.head(10))
