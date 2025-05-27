import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('cataract_model.pth', map_location=torch.device('cpu')))
model.eval()

# Classes
class_names = ['cataract', 'normal']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded image
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Open and transform
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            result = class_names[pred.item()]
            conf = confidence.item() * 100

        return render_template('index.html', result=result, conf=conf, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
