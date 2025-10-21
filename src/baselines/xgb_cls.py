from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __init__(self):
        super().__init__()