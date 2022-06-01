from utility import *
print("*__All Imported Successfully__*")

parser = argparse.ArgumentParser(description = "Prediction")
parser.add_argument('image', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--topk', type=int,default = 5)
parser.add_argument('--category_names',type=str, default = 'cat_to_name.json')
parser.add_argument('--cpu', type=str, default = 'cuda')
args = parser.parse_args()


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

#loading checkpoint
def loading_model(checkpoint_path):
    checkPointPath = torch.load(checkpoint_path) 
    
    #testing with vgg16
    arch = 'vgg16'
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size   = 25088
        hidden_units = 4096
        output_size  = 102
        
    elif arch == 'densenet161':
        model = models.densenet161(pretrained = True)
        input_size   = 1024
        hidden_units = 500
        output_size  = 102
    
    for pram in model.parameters():
        pram.requires_grad = False
        
    model.class_to_idx = checkPointPath['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size,hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, output_size)),
            ('output', nn.LogSoftmax(dim = 1))
]))

    model.classifier = classifier
    model.load_state_dict(checkPointPath['state_dict'])
    return model


#processing image
def process_image(image):
    img = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    image_transformed = transform(img)
    return np.array(image_transformed)

#making prediction
def predict(image,device ,model,cat_to_name ,topk=5):
    
    model.to(device)
    model.eval()
    
    t_image = torch.from_numpy(np.expand_dims(image,axis = 0)).type(torch.FloatTensor).to('cuda:0')
    
    l_prop = model.forward(t_image)
    lin_prop = torch.exp(l_prop)
    top_prop , top_labels = lin_prop.topk(topk)    
    top_prop = np.array(top_prop.detach())[0]
    top_labels = np.array(top_labels.detach())[0]    
    idx_to_class = {value: key for key , value in model.class_to_idx.items()}    
    top_labels = [idx_to_class[lab] for lab in top_labels]
    flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_prop , top_labels, flowers

#pinting predictiones
def probabilities(prop, flowers):
    for x, y in enumerate(zip(flowers, prop)):
        print("Index: {}".format(x+1),
             "Flower: {},   Result: {}%".format(y[0], round(y[1]*100)))

        
model = loading_model(args.checkpoint)
image_ten = process_image(args.image)
device  = (args.cpu)
top_prop , top_labels, flowers = predict(image_ten, device, model, cat_to_name, args.topk)
probabilities(top_prop, flowers)

