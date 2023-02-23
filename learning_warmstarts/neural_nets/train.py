import torch
import statistics
from learning_warmstarts.neural_nets.models import FFNet
from learning_warmstarts import problem_specs
from learning_warmstarts.dataset.obst_avoidance_dataset import ObstacleAvoidanceDataset, ObstacleAvoidanceBatchDataset
import wandb
from learning_warmstarts.neural_nets.loss_fns import ConstraintLoss, SimpleLagrangianLoss, LagrangianLoss, IpoptLoss, WeightedMSE
class ModelTrainer:
    
    def __init__(self, train_folder, test_folder, layers, batched, epochs, learning_rate, convergence_threshold, cost_fn, act_fn, load_duals, debug_mode):
        self.debug_mode = debug_mode
        self.batched = batched
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.convergence_threshold = float(convergence_threshold)
        self.shape = layers
        self.cost_fn = cost_fn
        self.act_fn = act_fn

        self.load_data(train_folder, test_folder, load_duals)

        self.train_setup()

        self.model_file = 'model.pt'

        config = dict (
            batched = self.batched,
            epochs = self.epochs,
            lr = self.learning_rate,
            shape = self.shape,
            activation = act_fn,
            cost_fn = cost_fn
        )

        if not self.debug_mode:
            self.run = wandb.init(project='learning_warmstarts', entity='laine-lab', config=config, job_type='train')

    def load_data(self, train_folder, test_folder, load_target_duals):
        if self.batched:
            train_data = ObstacleAvoidanceBatchDataset(train_folder, load_target_duals)
            test_data = ObstacleAvoidanceBatchDataset(test_folder, load_target_duals)

        else:        
            train_data = ObstacleAvoidanceDataset(train_folder,load_target_duals)
            test_data = ObstacleAvoidanceDataset(test_folder,load_target_duals)
        
        self.train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = None, batch_sampler=None)
        self.test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = None, batch_sampler=None)

    def train_setup(self):
        if self.act_fn == 'relu':
            self.model = FFNet(self.shape, torch.nn.functional.relu)
        elif self.act_fn == 'tanh':
            self.model = FFNet(self.shape, torch.nn.functional.tanh)
        elif self.act_fn == 'leaky_relu':
            self.model = FFNet(self.shape, torch.nn.functional.leaky_relu)    
        else:     
            raise ValueError('Specified activation function not supported')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        if self.cost_fn == 'mse':
            self.criterion = torch.nn.MSELoss(reduction="sum")
        elif self.cost_fn == 'wmse':
            self.criterion = WeightedMSE()
        elif self.cost_fn == 'custom':
            self.criterion = ConstraintLoss(problem_specs)
        elif self.cost_fn == 'lag':
            self.criterion = LagrangianLoss()
        elif self.cost_fn == 'slag':
            self.criterion = SimpleLagrangianLoss()
        elif self.cost_fn == 'ipopt':
            self.criterion = IpoptLoss(max_iters=25)
        else:
            raise ValueError('Specified cost function not supported')

    def run_train_loop(self):
        self.train()
        self.save_model() 

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        print('Saved Model')

        if not self.debug_mode:
            artifact = wandb.Artifact('trained_model', type='model')
            artifact.add_file(self.model_file)
        
            self.run.log_artifact(artifact)
        
    def train(self):
        """ Train a simple FFNet on the training data 
            using mini-batch or stochastic gradient descent  
        """
        print('Starting Training...')
    
        t = torch.FloatTensor([problem_specs['interval_duration']])

        train_losses = [0] * 10
        i = 0
        for epoch in range(self.epochs):
            train_iters = 0
            test_iters = 0
            moving_avg_train_loss = 0
            moving_avg_test_loss = 0

            self.model.train()
            total_train_loss = 0
            total_test_loss = 0

            total_train_samples = 0
            total_test_samples = 0
            for x, y in self.train_loader:
                if self.cost_fn == 'lag' or self.cost_fn == 'slag':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            p = torch.cat([x[r,:],t])
                            loss = self.criterion(prediction, y.float(),p)
                        loss /= x.shape[0]
                    else:                
                        prediction = self.model(x.float())
                        p = torch.cat([x,t])
                        loss = self.criterion(prediction, y.float(),p)
                
                elif self.cost_fn == 'mse' or self.cost_fn == 'wmse':
                    
                    prediction = self.model(x.float())
                    
                    loss = self.criterion(prediction, y.float())

                    total_train_loss += loss.item()
                    total_train_samples += y.shape[0]
               
                elif self.cost_fn == 'ipopt':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            loss += self.criterion(prediction, x[r,:])
                        
                        total_train_loss += loss.item()
                        loss /= x.shape[0]
                        total_train_samples += y.shape[0]

                    else:
                        prediction = self.model(x.float())
                        loss = self.criterion(prediction, x)

                        total_train_loss += loss.item()
                        total_train_samples += 1
                        
                elif self.cost_fn == 'custom':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            loss += self.criterion(prediction, y.float(), x[r,:])
                        loss /= x.shape[0]  
                    else:        
                        prediction = self.model(x.float())
                        loss = self.criterion(prediction, y.float(),x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                moving_avg_train_loss = (loss.item() + train_iters * moving_avg_train_loss) / (train_iters + 1)
                train_iters += 1

            self.model.eval()
            for x,y in self.test_loader:
                if self.cost_fn == 'lag' or self.cost_fn == 'slag':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            p = torch.cat([x[r,:],t])
                            loss += self.criterion(prediction, y.float(),p)
                        loss /= x.shape[0]
                    else:                
                        prediction = self.model(x.float())
                        p = torch.cat([x,t])
                        loss = self.criterion(prediction, y.float(),p).item()
                
                elif self.cost_fn == 'mse':
                    prediction = self.model(x.float())
                    loss = self.criterion(prediction, y.float())

                    total_test_loss += loss.item()
                    total_test_samples += y.shape[0]
                elif self.cost_fn == 'ipopt':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            loss += self.criterion(prediction, x[r,:])
                        
                        total_test_loss += loss.item()
                        loss /= x.shape[0]
                        total_test_samples += y.shape[0]                    
                    else:
                        prediction = self.model(x.float())
                        loss = self.criterion(prediction, x).item()
                        total_test_loss += loss
                        total_test_samples += 1 

                elif self.cost_fn == 'custom':
                    if self.batched:
                        loss = 0.0
                        for r in range(x.shape[0]):
                            prediction = self.model(x[r,:].float())
                            loss += self.criterion(prediction, y.float(), x[r,:])
                        loss /= x.shape[0]  
                    else:        
                        prediction = self.model(x.float())
                        loss = self.criterion(prediction, y.float(),x).item()

                moving_avg_test_loss = (loss + test_iters * moving_avg_test_loss) / (test_iters + 1)

                test_iters += 1

            train_loss = moving_avg_train_loss
            test_loss = moving_avg_test_loss
            train_loss = total_train_loss / total_train_samples
            test_loss = total_test_loss / total_test_samples
            train_losses[i % 10] = train_loss
            i += 1 
            if not self.debug_mode:
               wandb.log({"train loss": train_loss, "test loss": test_loss})
            
            print(f'Epoch {epoch} / {self.epochs}: \t Train Loss: {train_loss:.4e} \t Test Loss {test_loss:.4e}')

            # if(i > 9 and statistics.stdev(train_losses) < self.convergence_threshold * train_losses[(i-1) % 10]):
            #     print('Converged')
            #     break