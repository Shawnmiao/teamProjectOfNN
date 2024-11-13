import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('E:/centennial2024fall/unsurpvisedLearn/MingyueMiao/pythonProject/car_price_predictor/predictions/audi.csv')
print(df['model'].unique())
# Preprocess the data
df = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'])
X = df.drop('price', axis=1)
y = df['price']
print(df.columns)
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
class CarPricePredictor(nn.Module):
    def __init__(self):
        super(CarPricePredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = CarPricePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=1000):
    model.train()
    for epoch in range(epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("every step:{},loss:{}".format(epoch, loss.item()))
train_model(model, criterion, optimizer, X_train, y_train)

# Save the trained model and scaler
torch.save(model.state_dict(), 'car_price_predictor.pth')
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler trained and saved.")
