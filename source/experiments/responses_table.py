import pandas as pd
from experiments.responses_row import ResponsesRow

class ResponsesTable:
    def __init__(self):
        self.table = pd.DataFrame()

    def add_row(self, row: ResponsesRow):
        self.table = self.table._append(row.__dict__, ignore_index=True)

    def save(self, file_path: str):

        column_names = ResponsesRow().__dict__.keys()
        column_names = [column_name.replace("_", " ").title() for column_name in column_names]
        column_names = [column_name.replace(" Id", " ID") for column_name in column_names]

        if len(self.table.columns) == 0:
            self.table = pd.DataFrame(columns=column_names)

        self.table.columns = column_names
        self.table.to_csv(file_path, index=False)