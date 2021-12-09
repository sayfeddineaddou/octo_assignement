from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self,n_estimators = 120, max_depth: int = 4, criterion: str = 'entropy',class_weight=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        super().__init__(
            model=RandomForestClassifier(
                n_estimators = self.n_estimators,
                max_depth=self.max_depth,   
                criterion=self.criterion,
                class_weight = self.class_weight
            )
        )
