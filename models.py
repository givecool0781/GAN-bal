import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from os.path import join

SEED=2

class Softmax(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(Softmax,self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        #self.classifier = self.classifier'

    def forward(self,x):
        output= self.classifier(x)         
        return output


class CNN8(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(CNN8, self).__init__()
        # kernel
        self.input_dim = input_dim
        self.num_classes = num_classes

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,padding=1)) # ;input_dim,64
        conv_layers.append(nn.BatchNorm1d(8))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(16))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(32))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(64))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(256))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=256,out_channels=512,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(512))
        conv_layers.append(nn.ReLU(True))
        
        conv_layers.append(nn.Conv1d(in_channels=512,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))
        

        self.conv = nn.Sequential(*conv_layers).to(device)

        fc_layers = []
        fc_layers.append(nn.Linear(input_dim*128,num_classes))
        self.classifier = nn.Sequential(*fc_layers).to(device)

    def forward(self, x):
        batch_size, D = x.shape
        x = x.view(batch_size,1,D)

        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class CNN5(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(CNN5, self).__init__()
        # kernel
        self.input_dim = input_dim
        self.num_classes = num_classes

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,padding=1)) # ;input_dim,64
        conv_layers.append(nn.BatchNorm1d(64))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(256))
        conv_layers.append(nn.ReLU(True))
        
#         conv_layers.append(nn.Dropout(0.5))
        
        conv_layers.append(nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(256))
        conv_layers.append(nn.ReLU(True))

#         conv_layers.append(nn.Dropout(0.5))
        
        conv_layers.append(nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))
        
        self.conv = nn.Sequential(*conv_layers).to(device)

        fc_layers = []
        fc_layers.append(nn.Linear(input_dim*128,num_classes))
        self.classifier = nn.Sequential(*fc_layers).to(device)

    def forward(self, x):
        batch_size, D = x.shape
        x = x.view(batch_size,1,D)

        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class LSTM(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        layers = []
        layers.append(nn.LSTM(input_size=input_dim, hidden_size=80, num_layers=1, bidirectional=True, batch_first=True))

        self.LSTM = nn.Sequential(*layers).to(device)
        
        fc_layers = []        
        fc_layers.append(nn.Linear(160,num_classes))
        fc_layers.append(nn.Dropout(0.5))
        
        self.classifier = nn.Sequential(*fc_layers).to(device)

    def forward(self, x):
        batch_size, D = x.shape
#         print(x.shape)
        x = x.view(batch_size,1,D)
#         print(x.shape)
        r_out , (h_n, c_n) = self.LSTM(x)
#         print(r_out.shape)
        r_out = torch.flatten(r_out,1)
#         print(r_out.shape)
        x = self.classifier(r_out)

        return x

class CNN3(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(CNN3, self).__init__()
        # kernel
        self.input_dim = input_dim
        self.num_classes = num_classes

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1)) # ;input_dim,64
        conv_layers.append(nn.BatchNorm1d(16))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=16,out_channels=64,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(64))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))
        
        self.conv = nn.Sequential(*conv_layers).to(device)
        
        fc_layers = []
        fc_layers.append(nn.Linear(input_dim*128,num_classes))
        self.classifier = nn.Sequential(*fc_layers).to(device)

        
    def forward(self, x):
        batch_size, D = x.shape
        x = x.view(batch_size,1,D)

        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
        


class Net5(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(Net5, self).__init__()
        # kernel
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        layers.append(nn.Linear(input_dim,128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(128,256))
        
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256,256))
        
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.4))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256,128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU(True))        
        layers.append(nn.Linear(128,num_classes))

        self.model = nn.Sequential(*layers).to(device)
        
    def forward(self, x):
        return self.model(x)

class BAT(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(BAT, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        layers = []
        layers.append(nn.Conv1d(in_channels=1,out_channels=20,kernel_size=3,padding=1))
#         layers.append(nn.BatchNorm1d(20))
        #layers.append(nn.Dropout(0.5))
        layers.append(nn.Tanh())
        
        
        layers.append(nn.Conv1d(in_channels=20,out_channels=40,kernel_size=3,padding=1))
#         layers.append(nn.BatchNorm1d(40))
        layers.append(nn.Tanh())
        #layers.append(nn.Dropout(0.5))


        layers.append(nn.Conv1d(in_channels=40,out_channels=80,kernel_size=3,padding=1))
#         layers.append(nn.BatchNorm1d(80))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(0.5))

        layers.append(nn.LSTM(input_size=input_dim, hidden_size=80, num_layers=1, bidirectional=True, batch_first=True))
        

        
        self.bat = nn.Sequential(*layers).to(device)
        
        fc_layers = []        
        fc_layers.append(nn.Linear(12800,num_classes))
        fc_layers.append(nn.Dropout(0.5))
        
        self.classifier = nn.Sequential(*fc_layers).to(device)
        
    def forward(self, x):
        batch_size, D = x.shape
        
        x = x.view(batch_size,1,D)
        
        r_out , (h_n, c_n) = self.bat(x)
        r_out = torch.flatten(r_out,1)
        
        x = self.classifier(r_out)

        return x
        

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        #self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
            print(w)
        print("---------------------------------------------------")
        
        

class Classifier:
    def __init__(self,method,input_dim,num_classes,num_epochs,batch_size=100,lr=1e-3,reg=1e-5,runs_dir=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.reg= reg
        self.runs_dir = runs_dir
        self.device = 'cuda:0'

        #self.model = nn.Linear(self.input_size, self.num_classes).to(self.device)
        if method=='softmax':
            self.device = torch.device('cuda:0')
            self.model = Softmax(input_dim,num_classes=num_classes, device=self.device)
        elif method=='cnn8':
            self.device = torch.device('cuda:0')
            self.model = CNN8(input_dim,num_classes=num_classes,device=self.device)        
        elif method=='cnn5':
            self.device = torch.device('cuda:0')
            self.model = CNN5(input_dim,num_classes=num_classes,device=self.device)        

        elif method=='cnn3':
            self.device = torch.device('cuda:0')
            self.model = CNN3(input_dim,num_classes=num_classes,device=self.device)        
        elif method=='nn5':
            self.device = torch.device('cuda:0')
            self.model = Net5(input_dim,num_classes=num_classes,device=self.device)
        elif method=='BAT':
            self.device = torch.device('cuda:0')
            self.model = BAT(input_dim,num_classes=num_classes,device=self.device) 
        elif method=='LSTM':
            self.device = torch.device('cuda:0')
            self.model = LSTM(input_dim,num_classes=num_classes,device=self.device) 
        else:
            print('There is no such classifier')
            exit()
        self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, weight_decay=self.reg, amsgrad=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, amsgrad=False)

    def fit(self,X,Y):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        for dev_index, val_index in sss.split(X, Y): # runs only once
                X_dev = X[dev_index]
                Y_dev = Y[dev_index]
                X_val = X[val_index]
                Y_val = Y[val_index]  
        
        writer = SummaryWriter(self.runs_dir) 
        #Train data
        tensor_x = torch.stack([torch.Tensor(i) for i in X_dev]).to(self.device)
        tensor_y = torch.LongTensor(Y_dev).to(self.device) # checked working correctly
        dataset = utils.TensorDataset(tensor_x,tensor_y)        
        train_loader = utils.DataLoader(dataset,batch_size=self.batch_size)
        
        #Test data
        tensor_x_val = torch.stack([torch.Tensor(i) for i in X_val]).to(self.device)
        tensor_y_val = torch.LongTensor(Y_val).to(self.device)
        dataset_val = utils.TensorDataset(tensor_x_val,tensor_y_val)
        test_loader = utils.DataLoader(dataset_val,batch_size=self.batch_size)
        
        
        N = tensor_x.shape[0] #個數

        num_epochs = self.num_epochs

        model  = self.model
        best_acc = None
        best_epoch = None
        val_loss_min = 0
        weight_decay = 0 #權重衰減
       

        filepath = join(self.runs_dir,'checkpoint.pth')
        
        if os.path.isfile(filepath):
            checkpoint = self.load_checkpoint(filepath)
            best_epoch = checkpoint['epoch']
            best_batch = checkpoint['batch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            pred = self.predict(X_val)
            best_acc = metrics.balanced_accuracy_score(Y_val,pred)*100 #val acc
            resume_epoch = best_epoch+1  
            resume_batch = best_batch+1
        else:
            resume_epoch = 0
            resume_batch = 0
            best_acc = -1
            best_epoch = 0

        no_improvement = 0
        print("best epoch {}, best batch {}".format(resume_epoch,resume_batch))
        print("bst acc ", best_acc)
        
        for epoch in range(resume_epoch,num_epochs):
            for i,(xi,yi) in enumerate(train_loader):
                if epoch==resume_epoch and i<resume_batch:
                    continue
                model.train()
                
                outputs = model(xi)
                loss = self.criterion(outputs,yi)
                #權重
#                 if weight_decay>0:
#                     reg_loss=Regularization(model, weight_decay, p=0).to(self.device)
#                 else:
#                     print("no regularization")


#                 if weight_decay > 0:
#                     loss_r = loss + reg_loss(model)
               
                

                    
                loss.requires_grad
                seen_so_far = self.batch_size*(epoch*len(train_loader)+i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                
                
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                
                
                 
                if (seen_so_far/self.batch_size)%50==0:
                    
                    pred = self.predict(X_val) 
                    balanced_acc = metrics.balanced_accuracy_score(Y_val,pred)*100
                    
                    
                    val_loss = validation(self, model, test_loader)
                    if weight_decay > 0:
                        val_loss_r = val_loss + reg_loss(model)
                    if val_loss < val_loss_min:
                        val_loss_min=val_loss

                    if loss < val_loss_min:
                        best_acc = balanced_acc
                        best_epoch = epoch
                        checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch':epoch,
                        'batch': i,
                        'batch_size': self.batch_size
                        }
                        self.save(checkpoint)
                        if val_loss > val_loss_min:
                            no_improvement =0

                    else:
                        no_improvement+=1 #設置epoch停止線
                        if no_improvement>=10:
                            print("no improvement in accuracy for 10 iterations")
                            best_acc = balanced_acc
                            best_epoch = epoch
                            checkpoint = {
                            'state_dict': model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'epoch':epoch,
                            'batch': i,
                            'batch_size': self.batch_size
                            }
                            self.save(checkpoint)
                            return
                    
              
                    
                        
                    writer.add_scalars('train_loss/val_loss',{'train_loss':loss.item(),'val_loss':val_loss.item()}, epoch)
                    
                    acc = metrics.accuracy_score(Y_val,pred)*100
                    
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)
                        
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, val_Loss: {:.4f}' 
                                               .format(epoch+1, num_epochs, i+1, len(Y_dev)//self.batch_size, loss.item(), val_loss.item()))
        writer.close()
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : self.optimizer.state_dict(),
        'epoch':num_epochs,
        'batch': num_epochs,
        'batch_size': self.batch_size
        }
        self.save(checkpoint)       


    def predict(self,x,eval_mode=False):
        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(self.device)
        bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64).to(self.device)
        
        if eval_mode:
            model = self.load_model()
        else:
            model = self.model
        model.eval()        
        
        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                
                outputs = model(xi)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))

        return pred.cpu().numpy()


    def save(self,checkpoint):
        path = join(self.runs_dir,'checkpoint.pth')
        torch.save(checkpoint,path)

    
    def load_checkpoint(self,filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            print("Loaded {} model trained with batch_size = {}, seen {} epochs and {} mini batches".
                format(self.runs_dir,checkpoint['batch_size'],checkpoint['epoch'],checkpoint['batch'])) 
            return checkpoint
        else:
            return None
        
            
    def load_model(self,inference_mode=True):
        filepath = join(self.runs_dir,'checkpoint.pth')
        checkpoint = self.load_checkpoint(filepath)
        
        model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        
        if inference_mode:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        return model
    
    
def validation(self,model, test_loader): #early stop 用
    # Settings
    
    model.eval()
    loss_total = 0

    # Test validation data
    with torch.no_grad():
        for i,(xi_val,yi_val) in enumerate(test_loader):

            output_val = model(xi_val)
            val_loss = self.criterion(output_val,yi_val)

    return val_loss

