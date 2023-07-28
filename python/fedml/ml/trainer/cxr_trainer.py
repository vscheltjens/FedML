
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from ...core.alg_frame.client_trainer import ClientTrainer
from model.utils import MSLELoss, Metrics


class ModelTrainerCXR(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.NLLLoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        elif args.client_optimizer == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        elif args.client_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.classifier.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            last_loss = 0. ; running_loss = 0. 
            total = 0 ; correct = 0
            targets_acc = [] ; outputs_acc = []
            for i, batch in enumerate(train_data):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)

                target = target.type(torch.LongTensor).to(device)
                    
                #zero grad optimizer for every batch
                model.zero_grad() #test this
                optimizer.zero_grad()

                #run model
                outputs = model(inputs.float())
                outputs = outputs.to(device)
                
                #backpropagation and gradient calculation
                print(outputs.size(), target.size())
                loss = criterion(outputs, target) 
                loss.backward()

                #Adjust model weights
                optimizer.step()        
                
                # update running training loss
                running_loss += loss.item()
                last_loss = running_loss/(i+1)
                
                #Get all the metrics and not just loss
                total, correct, predicted, true = Metrics.clf_metrics(total, correct, target.cpu().detach(), output.cpu().detach())
                print(f'Correct: {correct}, total: {total}')
                acc = correct / total

                outputs_acc.append(predicted)
                targets_acc.append(true)

                #print some information to stdout
                print('Epoch {}, batch {} ------ train_loss: {} ------ average train_loss: {} ------ accumulated train_accuracy: {}'.format(epoch, i + 1, loss.item(), last_loss, acc)) 

            outputs_y = np.concatenate(outputs_acc)
            targets_y = np.concatenate(targets_acc)

            f1 = f1_score(targets_y, outputs_y, average='micro')
            acc_ref = accuracy_score(targets_y, outputs_y)

            #calculate the epoch level metrics
            epoch_metrics = [acc, acc_ref, f1, last_loss]
            print(f'epoch_metrics {epoch_metrics}')


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        last_loss = 0. ; running_loss = 0. 
        total = 0 ; correct = 0
        targets_acc = [] ; outputs_acc = []
        
        metrics = {"Acc": 0, "Acc_res": 0, "F1": 0, "Loss": 0}
        criterion = nn.NLLLoss().to(device)

        with torch.no_grad():
            for i, batch in enumerate(test_data):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)

                target = target.type(torch.LongTensor).to(device)

                #run model
                outputs = model(inputs.float())
                outputs = outputs.to(device)
                
                #backpropagation and gradient calculation
                loss = criterion(outputs, target) 
            
                # Gather data and report
                running_loss += loss.item()
                print(f'test loss {running_loss/(i+1)}')

                #Get all the metrics and not just loss
                total, correct, predicted, true = Metrics.clf_metrics(total, correct, target.cpu().detach(), outputs.cpu().detach())
                print(f'Correct: {correct}, total: {total}')
                acc = correct / total

                outputs_acc.append(predicted)
                targets_acc.append(true)

            outputs_y = np.concatenate(outputs_acc)
            targets_y = np.concatenate(targets_acc)

            f1 = f1_score(targets_y, outputs_y, average='micro')
            acc_ref = accuracy_score(targets_y, outputs_y)
            loss = running_loss/(i+1)

            #calculate the epoch level metrics
            test_metrics = [acc, acc_ref, f1, loss]
            
            
            for i, m in enumerate(metrics.keys()):
                metrics[m] = test_metrics[i]
                print(f' test metrics: {test_metrics}, {metrics}')

            return metrics