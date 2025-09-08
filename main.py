import os
import re
import json
from typing import List, Dict
from unittest import result
import emoji
from bs4 import BeautifulSoup as bs
import pandas as pd

train_dir = "./tweet/train"
test_dir = "./tweet/test"


class DataProcessing:
    def __init__(self):
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
        # Split by whitespace
        tokens = content.split()
        return tokens

    def data_cleaning(self, content: str) -> str:

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # remove emoji
        content = emoji.demojize(content)

        content = self.remain_capital_words(content)

        return content

    def build_vocabulary(self, label: int) -> set:
        """
        Build vocabulary from texts with specified label. And store the pandas DataFrame as csv file.
        """
        vocab = set()

        filtered_data = self.training[self.training["label"] == label].copy()
        filtered_data["tokens"] = filtered_data["content"].apply(
            self.tokenize_and_remove_punc
        )

        if label == 1:
            filtered_data.to_csv("positive_label_data.csv", index=False)
        else:
            filtered_data.to_csv("negative_label_data.csv", index=False)

        for tokens in filtered_data["tokens"]:
            vocab.update(tokens)

        return vocab

    def main(self) -> Dict[str, set]:

        result = dict()

        def build_dataframe(file_dir: str) -> pd.DataFrame:
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

        self.training = build_dataframe(train_dir)
        # self.testing = build_dataframe(test_dir)

        self.training["content"] = self.training["content"].apply(self.data_cleaning)
        # self.testing["content"] = self.testing["content"].apply(self.data_cleaning)

        positive_vocab = self.build_vocabulary(label=1)
        word2idx = {word: idx for idx, word in enumerate(positive_vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}

        result["positive_vocab"] = list(positive_vocab)
        result["word2idx"] = word2idx
        result["idx2word"] = idx2word

        with open("vocab.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result


if __name__ == "__main__":
    if not os.path.exists("positive_label_data.csv") or not os.path.exists(
        "vocab.json"
    ):
        processor = DataProcessing()
        result = processor.main()
    else:
        print(
            "positive_label_data.csv and vocab.json already exists. Skipping data processing."
        )
        positive_data = pd.read_csv("positive_label_data.csv")
        vocab = json.load(open("vocab.json", "r", encoding="utf-8"))
        print(positive_data.head())
