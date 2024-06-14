import pandas as pd
import csv
from models.response import Response
from logs.log import Log
from experiments.details_row import DetailsRow

class DetailsTable:

    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.table = pd.DataFrame()

    def create_header(self):
        column_names = DetailsRow().__dict__.keys()
        column_names = [column_name.replace("_", " ").title() for column_name in column_names]
        column_names = [column_name.replace(" Id", " ID") for column_name in column_names]
        self.writer.writerow(column_names)

    def create_row(self):
        return DetailsRow()

    def add_row(self, row):
        self.table = self.table._append(row.__dict__, ignore_index=True)
        self.writer.writerow(row.__dict__.values())

    def close(self):
        self.file.close()


    def update_details_row(self, details_row: DetailsRow, response: Response) -> None:
        details_row.input_tokens += response.input_tokens
        details_row.output_tokens += response.output_tokens
        details_row.total_tokens += response.total_tokens

        if (response.has_error):
            details_row.error += response.text

    def log_detail_summary(self, log: Log, details_row: DetailsRow):

        votes = []
        for i in range(1, 11):
            vote = getattr(details_row, f"answer_{i}")
            if vote is None:
                vote = "[None]"
            votes.append(vote)

        log.info(f"Votes: {', '.join(votes)}")
        log.info(f"Agent Answer: {details_row.agent_answer}")
        log.info(f"Correct Answer: {details_row.correct_answer}")
        log.info(f"Score: {details_row.score}")
        log.info(f"Tokens: {details_row.total_tokens}")
        log.info("")



