import os
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

    def data_cleaning(self, content: str) -> str:

        def remain_capital_words(content: str) -> str:
            # detect fully capitalized words
            words = content.split()
            return " ".join(
                [
                    word if sum(1 for ch in word if ch.isupper()) > 1 else word.lower()
                    for word in words
                ]
            )

        # remove HTML tags
        content = bs(content, "html.parser").get_text()

        # remove emoji
        content = emoji.demojize(content)

        return content

    def main(self):
        def build_dataframe(file_paths: List[str]) -> pd.DataFrame:
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

        self.training = build_dataframe(self.list_all_files(train_dir))
        self.testing = build_dataframe(self.list_all_files(test_dir))

        print(self.training.head())
        print(self.testing.head())


if __name__ == "__main__":
    processor = DataProcessing()
    processor.main()
