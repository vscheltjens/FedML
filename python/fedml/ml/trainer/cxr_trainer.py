
import torch
import numpy as np
import torch.nn as nn
import wandb
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
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=1e-3,
                weight_decay=1e-5,
                amsgrad=True,
            )

        criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.ones([args.num_classes])).to(device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.batch_size, T_mult=1)

        epoch_loss = []
        for epoch in range(args.epochs):
            last_loss = 0. ; running_loss = 0. 
            running_auc = 0.
            running_metrics = [0., 0., 0., 0., 0.]
            
            scheduler.step()

            for i, batch in enumerate(train_data):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)
                    
                #zero grad optimizer for every batch
                optimizer.zero_grad()

                #run model
                outputs = model(inputs)
                outputs = outputs.to(device)
                
                #backpropagation and gradient calculation
                loss = criterion(outputs, target) 
                loss.backward()

                #Adjust model weights
                optimizer.step()        
                
                # update running training loss
                running_loss += loss.item()
                last_loss = running_loss/(i+1)

                # print(f'Target info: {target.shape}, output info: {outputs.shape}')
                
                batch_auc, batch_label_auc = Metrics.auc_metrics(target, outputs)

                running_auc += batch_auc.item()
                running_metrics = [*map(sum, zip(running_metrics, batch_label_auc.tolist()))]


                if (i+1) % args.checkpoint == 0:
                    print('Epoch {}, batch {} ---------- train_loss: {} ---------- average train_loss: {} ---------- average train_AUC {}.'.format(epoch, i + 1, loss.item(), last_loss, running_auc/(i+1)))
                    wandb.log({
                        'batch_train_loss': last_loss,
                        'batch_train_auc': running_auc/(i+1),
                        }) 

                # if i == 10:
                #     break

            avg_auc = running_auc/(i+1)
            epoch_label_auc = [metric/(i+1) for metric in running_metrics]

            print(last_loss, avg_auc, epoch_label_auc)


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        last_loss = 0. ; running_loss = 0. 
        running_auc = 0.
        running_metrics = [0., 0., 0., 0., 0.]
        
        metrics = {"AUC1": 0, "AUC2": 0, "AUC3": 0, "AUC4": 0, "AUC5": 0}
        criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.ones([args.num_classes])).to(device)

        with torch.no_grad():
            for i, batch in enumerate(test_data):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)

                #run model
                outputs = model(inputs)
                outputs = outputs.to(device)
                
                #backpropagation and gradient calculation
                loss = criterion(outputs, target) 
            
                # Gather data and report
                running_loss += loss.item()
                # last_loss = running_loss/(i+1)
                # print(f'Target info: {target.shape}, output info: {outputs.shape}')

                batch_auc, batch_label_auc = Metrics.auc_metrics(target, outputs)

                running_auc += batch_auc.item()
                running_metrics = [*map(sum, zip(running_metrics, batch_label_auc.tolist()))]


                if (i+1) % args.checkpoint == 0:
                    #Print info to stdout
                    print('batch {} ---------- val_loss: {} ---------- average val_loss: {} ---------- average val_AUC {}.'.format( i + 1, loss.item(), last_loss, running_auc/(i+1)))
                    wandb.log({
                        'batch_val_loss': last_loss,
                        'batch_val_auc': running_auc/(i+1),
                        }) 

                # # if i == 5:
                # #     break

            avg_auc = running_auc/(i+1)
            epoch_label_auc = [metric/(i+1) for metric in running_metrics]

            print(last_loss, avg_auc, epoch_label_auc)
            
            
            for i, m in enumerate(metrics.keys()):
                metrics[m] = epoch_label_auc[i]
            
            print(f' test metrics: {epoch_label_auc}, {metrics}')

            return metrics