import wandb 
from sklearn.metrics import f1_score, classification_report

from utils.logger import get_logger 

logger=get_logger(__name__)

def eval(model, X_test, y_test,config):
    logger.info("Model evaluation started:")
    
    with wandb.init(
    project="student-drop-enroll-grad-preds",
        tags=["Eval", "F1-Score"]
    ) as run:
        pred = model.predict(X_test)

        score = f1_score(
            y_pred=pred,
            y_true=y_test,
            average='macro'
        )

        report = classification_report(
            y_pred=pred,
            y_true=y_test,
            target_names=config.preprocessing.target
        )

        run.log({
            "model_name": config.model.model_name,
            "f1_macro": score
           }
        )
    
    result = {
        "model_name": config.model.model_name,
        "score": score
    }
    logger.info("Model evaluation done")

    return result, report