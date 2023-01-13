# Import dependencies
import torch
import torch.nn as nn
from sklearn import metrics

#MSLE Loss and Metrics Class
class MSLELoss(nn.Module):
  def __init__(self):
      super().__init__()
      self.mse = nn.MSELoss()    
  def forward(self, true, pred):
      return self.mse(torch.log(true + 1), torch.log(pred + 1))


class Metrics():
  def labelling_outcomes():
    true_labels = 2
    pred_labels = 2
    return true_labels, pred_labels

  def diff_metrics(true, pred):
    mae = metrics.mean_absolute_error(true, pred)
    mape = metrics.mean_absolute_percentage_error(true, pred) #might have to perform the same trick like Rocheteau as mape can be very big for small ytrue values.
    mse = metrics.mean_squared_error(true, pred)
    msle = metrics.mean_squared_log_error(true, pred)#(true+1, pred+1)
    R_sq = metrics.r2_score(true, pred)
    return [mae, mape, mse, msle, R_sq]

