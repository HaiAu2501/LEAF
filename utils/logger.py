import os
import json

class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_to_json(self, data: dict, file_name):
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def log_to_txt(self, data: str, file_name: str):
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(data)