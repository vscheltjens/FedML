# Import dependencies
import os 
import json
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

def save_model(args, model, timestamp):
    stamp = 'training_{}'.format(timestamp)
    dir_path = os.path.join(args.model_dir, stamp)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # arg_path = os.path.join(dir_path + '/' + 'args.json')

    # if not os.path.exists(arg_path):
    #     with open(arg_path, 'w') as fp:
    #         json.dump(args, fp, sort_keys = True, indent = 4)

    model_p = 'model_federated'
    torch.save(model.state_dict(), dir_path +'/'+ model_p) 

