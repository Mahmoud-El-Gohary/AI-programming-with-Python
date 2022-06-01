from utility import *
print("*__All Imported Successfully__*")
  
    
    
#defining arguments.
parser = argparse.ArgumentParser(description = "Training The Network")
parser.add_argument('data_dir')
parser.add_argument('--save_dir', type=str, default = 'checkpoint.pth')
parser.add_argument('--arch', type=str ,default = 'vgg16')
parser.add_argument('--learning_rate',type = float ,default = 0.001)
parser.add_argument('--hidden_units', type = int, default = 4096)
parser.add_argument('--epochs',type = int, default = 25)
parser.add_argument('--cpu', type = str,default = 'cuda')
args = parser.parse_args()


print("Data Dirctory: {}".format(args.data_dir))
print("Saving Directory: {}".format(args.save_dir))
print("Model: {}".format(args.arch))
print("Learning Rate: {}".format(args.learning_rate))
print("Hidden Layers: {}".format(args.hidden_units))
print("Number Of Epochs: {}".format(args.epochs))
print("GPU Or CPU: {}".format(args.cpu))

#assign input_size, output_size depending on the selected model.
if (args.arch == 'vgg16'):
    input_size = 25088
    output_size = 102
elif (args.arch == 'densenet161'):
    input_size = 1024
    output_size =102

#defing loaders
train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle = True) 
valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 32, shuffle = True) 
test_loader  = torch.utils.data.DataLoader(image_datasets['test'],  batch_size = 32, shuffle = True) 

#assign the defined loaders to a dict
dataloaders  = {
    'train' : train_loader, 
    'valid' : valid_loader,
    'test'  : test_loader
}

#Downloading models
if (args.arch == 'vgg16'):
    model = models.vgg16(pretrained = True)
elif (args.arch == 'densenet161'):
    model = models.densenet161(pretrained = True)
    
#Defining our network   
for pram in model.parameters():
    pram.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size,args.hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(args.hidden_units, output_size)),
    ('output', nn.LogSoftmax(dim = 1))
]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr =float(args.learning_rate))


#Training Network
model.to(args.cpu)
print('Start Training....')
for epoch in range(args.epochs):
    for dataset in ['train', 'valid']:
        if dataset == 'train':
            model.train()
        else:
            model.eval()
        
        changing_loss, changing_accuracy = 0.0, 0.0
        
        for inputs , labels in dataloaders[dataset]:
            inputs, labels = inputs.to(args.cpu), labels.to(args.cpu)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(dataset == 'train'):
                output  = model(inputs)
                _, pred = torch.max(output, 1)
                loss    = criterion(output, labels)
            
                if dataset == 'train':
                    loss.backward()
                    optimizer.step()
                    
            changing_loss     +=  loss.item() * inputs.size(0)
            changing_accuracy += torch.sum(pred == labels.data)
            
        dataset_size = {
            x: len(image_datasets[x])
            for x in ['train', 'valid', 'test']
        }
        
        epoch_loss = changing_loss / dataset_size[dataset]
        epoch_accuracy = changing_accuracy.double() / dataset_size[dataset]
        
        print("Epoch: {}/{}\n".format(epoch+1, args.epochs),
             "{} Loss: {:.4f}\t Accuracy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))

        
#define a function the calclute accuracy
def check_acc(test_loader):
    correct , total = 0, 0
    model.to(args.cpu)

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(args.cpu), labels.to(args.cpu)
            output = model(images)
            _, pred = torch.max(output.data, 1 )
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
        print("Model Accuracy: %d%%" % (100 * correct / total))       
        
#using the model to check the model accuracy        
check_acc(dataloaders['train'])

#saving checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save(
    {
    'model'        : args.arch,
    'state_dict'   : model.state_dict(),
    'class_to_idx' : model.class_to_idx
    },
args.save_dir
)
print("Saving Model To: {}".format(args.save_dir))