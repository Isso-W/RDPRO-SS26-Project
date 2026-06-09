import json
from datetime import datetime


class SummaryGenerator:

    def save_report(
        self,
        report,
        dataset_name,
        filename
    ):

        report["dataset_info"] = {

            "name": dataset_name,

            "source": "huggingface",

            "analysis_timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print(
            f"Report saved to {filename}"
        )