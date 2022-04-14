from os.path import exists
from os import mkdir
import numpy as np
from tqdm.notebook import tqdm
from loss import enet_loss
import torch


def fit_enet(epochs, lr, model, train_loader,
             val_loader, device, checkpoints_path, save_interval,
             enet_checkpoint=None, opt_func=torch.optim.SGD):
    # initialize
    history, curr_epoch = [], 0
    optimizer = opt_func(model.parameters(), lr)

    # load checkpoint
    if exists(enet_checkpoint):
        checkpoint = torch.load(enet_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        curr_epoch = checkpoint['epoch']
        print('Checkpoint loaded')

    curr_state = evaluate_enet(model, val_loader, device)
    print(f'Current model state (epoch {curr_epoch}) | Loss: {curr_state:.5f}')

    # main loop
    for epoch in range(curr_epoch + 1, curr_epoch + epochs):

        # Training Phase
        model.train()
        for inputs, labels_bin, labels_inst in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            labels_bin = labels_bin.to(device)
            labels_inst = labels_inst.to(device)
            embeddings, logit = model(inputs)
            loss = enet_loss(logit, embeddings, labels_bin, labels_inst)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate_enet(model, val_loader, device)
        print(f'Epoch {epoch} | Loss: {result:.5f}')
        history.append(result)

        # save checkpoint
        if epoch % save_interval == 0:
            if not exists(checkpoints_path):
                mkdir(checkpoints_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, enet_checkpoint)

    return history


def evaluate_enet(model, val_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels_bin, labels_inst in tqdm(val_loader, leave=False):
            inputs = inputs.to(device)
            labels_bin = labels_bin.to(device)
            labels_inst = labels_inst.to(device)
            embeddings, logit = model(inputs)
            loss = enet_loss(logit, embeddings, labels_bin, labels_inst)
            losses.append(loss.cpu().numpy().item())

    return np.mean(losses)
