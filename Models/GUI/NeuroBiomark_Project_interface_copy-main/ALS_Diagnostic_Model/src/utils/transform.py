from torchvision import transforms

def transform_function(mean = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]):
    return transforms.Compose([
        transforms.Resize((400,400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])