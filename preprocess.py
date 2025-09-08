import os
import re
from typing import List, Dict
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

    def tokenize_and_remove_punc(self, content: str) -> list[str]:
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

    def main(self) -> Dict[str, set]:
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

        # print(self.training.head())
        # print(self.testing.head())

        positive_vocub = set()
        self.training[self.training["label"] == 1]["content"].apply(
            lambda x: positive_vocub.update(self.tokenize_and_remove_punc(x))
        )
        return {"positive_vocub": positive_vocub}


if __name__ == "__main__":
    processor = DataProcessing()
    result = processor.main()
    print(result)
