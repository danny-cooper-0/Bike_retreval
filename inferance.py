import requests
import torch
from PIL import Image
import numpy as np
import io
import smtplib
from email.mime.text import MIMEText
import torch.nn as nn

# CNN model definition 
class CNNFull(nn.Module):
    def __init__(self, n_classes):
        super(CNNFull, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1_input_size = 128 * 28 * 28  # Adjust for 224x224 input images
        self.fc1 = nn.Linear(self.fc1_input_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

# Function to send email notifications
def send_email(subject, body, to_email):
    from_email = '' # Email and password cleared for security reasons
    password = ''

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

# Function to search eBay listings for black bikes
def search_ebay(query):
    app_id = '' # App ID cleared for security reseons
    url = 'https://svcs.ebay.com/services/search/FindingService/v1'
    params = {
        'OPERATION-NAME': 'findItemsByKeywords',
        'SERVICE-VERSION': '1.0.0',
        'SECURITY-APPNAME': app_id,
        'RESPONSE-DATA-FORMAT': 'JSON',
        'keywords': query,
        'paginationInput.entriesPerPage': 10
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to preprocess eBay image
def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W)
    return torch.tensor(img).unsqueeze(0)  # Add batch dimension

# Function to check if the eBay listing is a similar bike
def check_bike_similarity(image_url, model):
    img_tensor = preprocess_image(image_url)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    return output.item() > 0.5  # 0.5 threshold for binary classification

# Main function to load model, check eBay and notify if similar bike is found
def check_ebay_and_notify():
    model = CNNFull(n_classes=1)

    # Load saved model checkpoint
    model.load_state_dict(torch.load('cnn_checkpoint.pth'))

    listings = search_ebay('black bike')

    email_body = ""
    similar_bike_found = False

    for item in listings['findItemsByKeywordsResponse'][0]['searchResult'][0]['item']:
        title = item['title']
        price = item['sellingStatus'][0]['currentPrice'][0]['__value__']
        item_url = item['viewItemURL']
        image_url = item['galleryURL']

        if check_bike_similarity(image_url, model):
            email_body += f"Similar bike found: {title}\nPrice: ${price}\nLink: {item_url}\n\n"
            similar_bike_found = True

    if similar_bike_found:
        send_email(
            subject="Similar Bike Found on eBay",
            body=email_body,
            to_email= "" # Email cleard for privacy reasons
        )
        print("Email sent with similar bike listings.")

if __name__ == "__main__":
    # Run the inference script to check eBay and notify if similar bike is found
    check_ebay_and_notify()
