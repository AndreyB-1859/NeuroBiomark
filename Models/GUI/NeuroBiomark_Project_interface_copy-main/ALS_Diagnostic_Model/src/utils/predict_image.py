from PIL import Image
import torch
from ALS_Diagnostic_Model.src.utils.load_model import load_model
from ALS_Diagnostic_Model.src.utils.transform import transform_function

def predict_image(model, image_path):

    transform = transform_function()
    label_map = {0: "Control", 1: "Concordant", 2: "Discordant"}

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).squeeze().tolist()
        class_idx = int(torch.argmax(output))
        predicted_class = label_map[class_idx]
    return predicted_class, probs, image_tensor, class_idx


if __name__ == "__main__":

    weights_path = r"ALS_Diagnostic_Model/src/model/fold_0_model_weights.pth"
    model = load_model(weights_path)

    image_path = r"/Home_Window/4.tif"
    predicted_class, probs = predict_image(model, image_path)
    print(predicted_class, probs)