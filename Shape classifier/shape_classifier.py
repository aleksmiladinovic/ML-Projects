import numpy as np
from matplotlib import pyplot as plt
import pickle

from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

import torch
from torch import nn
from torch.nn import functional
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


np.random.seed(25)
torch.manual_seed(25)


def get_device():
    if torch.cuda.is_available():
        d = 'cuda'
    else:
        d = 'cpu'
    return torch.device(d)


def bind_gpu(data):
    d = get_device()
    if isinstance(data, (list, tuple)):
        return [bind_gpu(data_elem) for data_elem in data]
    return data.to(d, non_blocking=True)


def read_data():
    x = []
    y = []
    imgs = []
    for i in range(8):
        with open('data'+str(i)+'.aca', 'rb') as f, open('data description'+str(i)+'.aca', 'rb') as fd:
            n = pickle.load(f)
            imgs = pickle.load(f)
            for j in range(int(n/2)):
                x.append(pickle.load(f))
                y.append(pickle.load(fd))
    l = len(x)
    x = np.asarray([255*i for i in x]).astype('uint8')
    x = x.reshape(l, imgs[0], imgs[1], imgs[2])
    x = x[:, :, :, 0]
    return x, np.array(y)


def transform_categorical(y):
    categorify = {'rectangle': [1.0, 0.0, 0.0], 'circle': [0.0, 1.0, 0.0], 'triangle': [0.0, 0.0, 1.0]}
    yc = []
    for shape in y:
        yc.append(categorify[shape])

    return np.array(yc)


def transform_data(x, y):
    device = get_device()
    y = transform_categorical(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, stratify=y, random_state=25)

    shape_train = x_train.shape
    shape_test = x_test.shape
    x_train = x_train.reshape(shape_train[0], shape_train[1] * shape_train[2])
    x_test = x_test.reshape(shape_test[0], shape_test[1] * shape_test[2])
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype('float32')
    x_test = scaler.transform(x_test).astype('float32')
    x_train = x_train.reshape(shape_train[0], 1, shape_train[1], shape_train[2])
    x_test = x_test.reshape(shape_test[0], 1, shape_test[1], shape_test[2])

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=25)

    train_data = TensorDataset(torch.tensor(x_train).to(device), torch.tensor(y_train).to(device))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_data = TensorDataset(torch.tensor(x_val).to(device), torch.tensor(y_val).to(device))
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

    test_data = TensorDataset(torch.tensor(x_test).to(device), torch.tensor(y_test).to(device))
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader


x, y = read_data()
train_loader, val_loader, test_loader = transform_data(x, y)


class ShapeClassifier(nn.Module):
    def __init__(self, num_of_classes=3):
        super(ShapeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=10560, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_of_classes)

    def forward(self, a):
        a = self.conv1(a)
        a = functional.relu(a)
        a = self.conv2(a)
        a = functional.relu(a)
        a = functional.max_pool2d(input=a, kernel_size=2)
        a = self.dropout1(a)
        a = torch.flatten(a, 1)
        a = self.fc1(a)
        a = functional.relu(a)
        a = self.dropout2(a)
        a = self.fc2(a)
        output = functional.log_softmax(a, dim=1)
        return output


num_of_classes = 3
model = ShapeClassifier(num_of_classes=num_of_classes)
bind_gpu(model)


def train_classifier(model, criterion, optimizer, num_of_epochs, train_loader, val_loader):
    device = get_device()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_len = len(train_loader)
    val_len = len(val_loader)

    for epoch in range(num_of_epochs):
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()*batch_size
            predicted = torch.argmax(outputs.squeeze(), dim=1)
            labels = torch.argmax(labels, dim=1)
            train_correct += (predicted.squeeze() == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.size(0)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                predicted = torch.argmax(outputs.squeeze(), dim=1)
                labels = torch.argmax(labels, dim=1)
                val_running_loss += loss.item()*batch_size
                val_correct += (predicted.squeeze() == labels).sum().item()
                val_total += labels.size(0)

        train_epoch_loss = train_running_loss / train_len
        train_epoch_accuracy = train_correct / train_total
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)

        val_epoch_loss = val_running_loss / val_len
        val_epoch_accuracy = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

    for epoch in range(num_of_epochs):
        print(f"Epoch [{epoch + 1}/{num_of_epochs}], Train loss: {train_losses[epoch]:.4f}, Train accuracy: {train_accuracies[epoch]:.4f}")

    for epoch in range(num_of_epochs):
        print(f"Epoch [{epoch + 1}/{num_of_epochs}], Validation loss: {val_losses[epoch]:.4f}, Validation accuracy: {val_accuracies[epoch]:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


num_of_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

tlosses, taccuracies, vlosses, vaccuracies = train_classifier(model, criterion, optimizer, num_of_epochs, train_loader, val_loader)


def plot_classification(train_loss, train_accuracies, val_loss, val_accuracies):
    train_epochs = len(train_loss)
    val_epochs = len(val_loss)

    plt1 = plt.subplot2grid((10, 10), (0, 0), colspan=5, rowspan=5)
    plt2 = plt.subplot2grid((10, 10), (0, 7), colspan=5, rowspan=5)

    plt1.set_title('Loss over epochs')
    plt1.plot(np.arange(1, train_epochs+1), train_loss, color='b', label='train set')
    plt1.plot(np.arange(1, val_epochs+1), val_loss, color='r', label='validation set')
    plt1.set_xlabel('Epochs')
    plt1.set_ylabel('Loss')

    plt2.set_title('Accuracy over epochs')
    plt2.plot(np.arange(1, train_epochs+1), train_accuracies, color='b', label='train set')
    plt2.plot(np.arange(1, val_epochs+1), val_accuracies, color='r', label='validation set')
    plt2.set_xlabel('Epochs')
    plt2.set_ylabel('Accuracy')

    plt.show()


plot_classification(tlosses, taccuracies, vlosses, vaccuracies)


def save_model(model, fajl):
    torch.save(model.state_dict(), fajl)


def load_model(fajl):
    model = ShapeClassifier()
    model.load_state_dict(torch.load(fajl, weights_only=True))
    model.eval()
    return model


save_model(model, 'model.pt')
model = load_model('model.pt')


def evaluate_model(model, data_loader):
    model.eval()
    device = get_device()
    y_test = np.array([])
    y_predicted = np.array([])
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            originals = torch.argmax(labels, dim=1)
            #originals = labels.squeeze()
            y_test = np.append(y_test, originals.numpy())
            output = model(inputs)
            predicted = torch.argmax(output.squeeze(), dim=1)
            #predicted = output.squeeze()
            y_predicted = np.append(y_predicted, predicted.numpy())

    print(metrics.classification_report(y_test, y_predicted))

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_predicted, average='weighted')
    #print(f'Model evaluation on: {loader}')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    print(metrics.confusion_matrix(y_test, y_predicted))


print('Evaluation at the train set:')
evaluate_model(model, train_loader)

print('Evaluation at the test set')
evaluate_model(model, test_loader)

