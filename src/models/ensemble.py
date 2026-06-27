import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_2d import SimpleCNN
from models.resnet_transfer import model as resnet_model

class EnsembleModel:
    def __init__(self, models):
        self.models = models
        for m in self.models:
            m.eval()

    def predict_average(self, input_tensor):
        with torch.no_grad():
          predictions=[model(input_tensor) for model in self.models]
        return torch.stack(predictions).mean(dim=0)

        pass

    def predict_majority_vote(self,input_tensor,threshold=0.5):
        average_probability=self.predict_average(input_tensor)
        binary_prediction=(average_probability>threshold).float()
        return binary_prediction
        pass

    def predict_with_uncertainity(self, input_tensor):
        with torch.no_grad():
            predictions=torch.stack(
                [model(input_tensor) for model in self.models]
            )
        mean_pred=predictions.mean(dim=0)
        uncertainity=predictions.std(dim=0)
        return mean_pred, uncertainity
    
if __name__=="__main__":
    model1=SimpleCNN()
    model2=SimpleCNN()
    model3=SimpleCNN()

    ensemble = EnsembleModel([model1, model2, model3])
    fake_input=torch.randn(1,1,32,32)
    avg_pred=ensemble.predict_average(fake_input)
    majority=ensemble.predict_majority_vote(fake_input)
    mean, uncertainity=ensemble.predict_with_uncertainity(fake_input)

    print(f"Model 1 prediction: {model1(fake_input).item():.4f}")
    print(f"Model 2 prediction: {model2(fake_input).item():.4f}")
    print(f"Model 3 prediction: {model3(fake_input).item():.4f}")
    print(f"Ensemble average:   {avg_pred.item():.4f}")
    print(f"Majority vote:      {majority.item():.0f} (1=nodule, 0=not)")
    print(f"Uncertainty (std):  {uncertainity.item():.4f}")
    print()

    if uncertainity.item()>0.1:
        print(" high uncertainity-model disagree!")
        print("Recommend: radiologist review")
    else:
        print("low uncertainity-model agree!")
        print("confident prediction")

