from os import mkdir
from os.path import exists
import numpy as np
from tqdm.notebook import tqdm
from loss import compute_hnet_loss
import torch


def fit_hnet(epochs, lr, model, train_loader,
             val_loader, device, checkpoints_path, save_interval,
             order=2, hnet_checkpoint=None, opt_func=torch.optim.SGD):
    # initialize
    history, curr_epoch = [], 0
    optimizer = opt_func(model.parameters(), lr)

    # load checkpoint
    if exists(hnet_checkpoint):
        checkpoint = torch.load(hnet_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        curr_epoch = checkpoint['epoch']
        print('Checkpoint loaded')

    curr_state = evaluate_hnet(model, val_loader, order, device)
    print(f'Current model state | Loss: {curr_state:.5f}')

    # main loop
    for epoch in range(curr_epoch + 1, curr_epoch + epochs):

        # Training Phase
        model.train()
        for inputs, points in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            points = points.to(device)
            coefs = model(inputs)
            loss = compute_hnet_loss(points, coefs, order)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate_hnet(model, val_loader, order, device)
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
            }, hnet_checkpoint)

    return history


def evaluate_hnet(model, val_loader, order, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, points in tqdm(val_loader, leave=False):
            inputs = inputs.to(device)
            points = points.to(device)
            coefs = model(inputs)
            loss = compute_hnet_loss(points, coefs, order)
            losses.append(loss.cpu().numpy().item())

    return np.mean(losses)
