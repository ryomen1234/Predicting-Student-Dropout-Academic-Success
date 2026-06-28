import wandb 
from catboost import CatBoostClassifier

def train(X_train, y_train, config):
    params = config.model.best_params 
    
    model = CatBoostClassifier(**params)

    model.fit(X_train, y_train)

    return model

       





        
