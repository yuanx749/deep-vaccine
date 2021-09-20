import random
import os
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from data import Tokenizer, Tokenizer2, BaselineDataset, EpitopeDataset, AntigenDataset
from model import RNN
from utils import predict, plot_roc_curve

RANDOM_SEED = 42
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_eval(model, dataloader, criterion, optimizer=None, scheduler=None, is_train=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(is_train)
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_correct = 0
    progress_bar = tqdm(dataloader, ascii=True)

    for batch_idx, batch in enumerate(progress_bar):
        
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * len(labels)
        progress_bar.set_description_str('Batch: {:d}, Loss: {:.4f}'.format((batch_idx+1), loss.item()))

        predictions = torch.argmax(outputs, dim=1)
        total_correct += torch.sum(predictions.eq(labels))

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)

def run_experiment(hparams, epochs=50):

    tokenizer = Tokenizer(max_len=40)
    train_dataset_ = EpitopeDataset('Positive_train.txt', 'Negative_train.txt', tokenizer=tokenizer, data_dir='./data')
    test_dataset = EpitopeDataset('Positive_test.txt', 'Negative_test.txt', tokenizer=tokenizer, data_dir='./data')
    valid_size = 1000
    train_dataset, valid_dataset = random_split(train_dataset_, [len(train_dataset_) - valid_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=train_dataset.dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=valid_dataset.dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNN(tokenizer, emb_size=hparams['emb_size'], kernel_size=hparams['kernel_size'], hidden_size=hparams['hidden_size'], dropout=hparams['dropout'], pooling=hparams['pooling']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'], weight_decay=hparams['l2'], amsgrad=True)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 30], gamma=0.5)
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler4 = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.0001, hparams['lr'], step_size_up=100, cycle_momentum=False)
    criterion = nn.CrossEntropyLoss()

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    model_path = os.path.join(log_dir, 'lstm.pt')

    best_valid_loss = 0
    best_valid_acc = 0
    for epoch_idx in range(epochs):
        train_loss, train_acc = train_eval(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = train_eval(model, valid_loader, criterion, is_train=False)
        scheduler2.step()
        
        print('Epoch {}'.format(epoch_idx))
        print('Training Loss: {:.4f}. Valid Loss: {:.4f}. '.format(train_loss, valid_loss))
        print('Training Accuracy: {:.4f}. Valid Accuracy: {:.4f}. '.format(train_acc, valid_acc))
        writer.add_scalars('Loss', {'train': train_loss, 'Valid': valid_loss}, epoch_idx)
        writer.add_scalars('Accuracy', {'train': train_acc, 'Valid': valid_acc}, epoch_idx)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_loss = valid_loss
            torch.save(model, model_path)
    
    model = torch.load(model_path)
    _, test_acc = train_eval(model, test_loader, criterion, is_train=False)
    print("Test Accuracy: {:.4f}. ".format(test_acc))
    writer.add_hparams(hparams, {'hparam/accuracy': test_acc})
    
    inputs, labels = next(iter(train_loader))
    inputs = inputs.to(device)
    model = model.to(device)
    writer.add_graph(model, inputs)
    
    train_labels, train_probs = predict(model, train_loader)
    test_labels, test_probs = predict(model, test_loader)
    figure = plot_roc_curve(train_labels, train_probs, test_labels, test_probs)
    writer.add_figure('ROC', figure)
    writer.close()

if __name__ == '__main__':

    hparams = {
        'lr': 0.005,
        'l2': 0.0001,
        'batch_size': 1024,
        'emb_size': 32,
        'kernel_size': 5,
        'hidden_size': 128,
        'dropout': 0.2,
        'pooling': True
    }
    
    set_seed(RANDOM_SEED)
    run_experiment(hparams, epochs=50)
