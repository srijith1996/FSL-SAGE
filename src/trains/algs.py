# ------------------------------------------------------------------------------
import logging
import copy
import torch
from torch import optim

# ------------------------------------------------------------------------------
def aggregate_models(model_list, weights, device='cpu'):
    '''Aggregate Models
    
    Aggregate a given list of models using weights supplied in `weights`.

    Params
    ------
    model_list - List of pytorch models
    weights - List of weights used in averaging
    device - Device to use for computations

    Returns
    -------
    aggregated - Torch model containing the aggregated weights.
    '''

    assert len(weights) == len(model_list),\
        "Length of model_list and weights is different."

    assert sum(weights) == 1, "Sum of weights should be 1"

    aggregated = copy.deepcopy(model_list[0])
    aggregated_weights = aggregated.state_dict()

    for key in aggregated_weights:
        aggregated_weights[key] = model_list[0].state_dict()[key] * weights[0]

    for i in range(1, len(model_list)):
        for key in aggregated_weights:
            aggregated_weights[key] += model_list[i].state_dict()[key] * weights[i]

    aggregated.to(device)
    aggregated.load_state_dict(aggregated_weights)

    return aggregated

# ------------------------------------------------------------------------------
def fed_avg(
    rounds, iters, model, criterion, train_loader_list, test_loader, factor,
    lr, use_64bit=False, device='cpu'
):
    '''Federated Averaging

    Implements the FedAvg algorithm given a model.  Training happens for
    `rounds` rounds, with `iters` local iterations per client.

    Params
    ------
    rounds      - # training rounds
    iters       - # local iterations on each client, per round
    model       - PyTorch model to train
    train_loader_list - List of DataLoader objects, one for each client's
        training set.
    test_loader - DataLoader object for test data
    factor      - Weights used in model aggregation
    lr          - Learning rate
    use_64bit   - [optional] Use 64 bit precision
    device      - [optional] CPU or GPU to use for computation

    Returns
    -------
    client_copy_list - List of all Client() objects after training
    aggregated_client - Aggregated PyTorch client model
    server           - Server() object after training
    test_loss        - Loss values on test data during training
    acc              - Accuracy values on test data during training

    '''

    # copy client models
    num_clients = len(train_loader_list)
    clients = [copy.deepcopy(model).to(device) for _ in range(num_clients)]
    aggregated_client = copy.deepcopy(model).to(device)

    it_list = [iter(tl) for tl in train_loader_list]
    num_resets = [0 for _ in range(num_clients)]

    model_optim = [optim.Adam(c.parameters(), lr=lr) for c in clients]

    for r in range(rounds):
        for i in range(num_clients):
            for k in range(iters):

                # check if data iterator has finished iterating current cycle
                if (r * iters + k) == (num_resets[i] + 1) * len(it_list[i]):
                    num_resets[i] += 1
                    it_list[i] = iter(train_loader_list[i])

                samples, labels = next(it_list[i])
                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                
                model_optim[i].zero_grad()
                out = clients[i](samples)
                loss = criterion(out, labels)
                loss.backward()
                model_optim[i].step()

        # aggregate client models
        aggregated_client = aggregate_models(clients, factor, device=device)
        aggregated_client_weights = aggregated_client.state_dict()

        # Update client model weights
        for i in range(num_clients):
            clients[i].load_state_dict(aggregated_client_weights)
            
        # inference
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for samples, labels in test_loader:
                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                output = aggregated_client(samples)
                batch_loss = criterion(output, labels)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(output.data, 1)
                test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            acc =  100. * test_correct / len(test_loader.dataset)
            logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {acc:.2f}%')

    return aggregated_client, test_loss, acc

# ------------------------------------------------------------------------------
def sl_single_server(
    rounds, iters, client_copy_list, server, train_loader_list, test_loader,
    factor, client_lr, server_lr, use_64bit=False, device='cpu'
):
    '''Split Learning with single server model

    Implements split learning given a list of client models and a server model.
    Training happens for `rounds` rounds, with `iters` local iterations per
    client.

    Params
    ------
    rounds      - # training rounds
    iters       - # local iterations on each client, per round
    client_copy_list - list of Client() objects representing client model,
        optimizer and other options
    server      - Server() object representing the server model, optimizer and other
        options
    train_loader_list - List of DataLoader objects, one for each client's
        training set.
    test_loader - DataLoader object for test data
    factor      - Weights used in model aggregation
    client_lr, server_lr - Learning rates for client and server
    use_64bit   - [optional] Use 64 bit precision
    device      - [optional] CPU or GPU to use for computation

    Returns
    -------
    client_copy_list - List of all Client() objects after training
    aggregated_client - Aggregated PyTorch client model
    server           - Server() object after training
    test_loss        - Loss values on test data during training
    acc              - Accuracy values on test data during training

    '''
    logging.info(f"Configured rounds = {rounds}")

    num_clients = len(client_copy_list)
    client_optim_ws = [optim.Adam(c.model.parameters(), lr=client_lr) for c in client_copy_list]
    server_optim_ws = optim.Adam(server.model.parameters(), lr=server_lr)

    it_list = [iter(tl) for tl in train_loader_list]
    num_resets = [0 for _ in range(num_clients)]

    for r in range(rounds):
        for i in range(num_clients):
            for k in range(iters):

                # check if data iterator has finished iterating current cycle
                if (r * iters + k) == (num_resets[i] + 1) * len(it_list[i]):
                    num_resets[i] += 1
                    it_list[i] = iter(train_loader_list[i])

                # client feedforward
                samples, labels = next(it_list[i])
                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                client_optim_ws[i].zero_grad()
                server_optim_ws.zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)
                output = server.model(splitting_output) 
                loss = server.criterion(output, labels)
                loss.backward()
                server_optim_ws.step()
                client_optim_ws[i].step()

        # aggregate client model
        aggregated_client = aggregate_models(
            [c.model for c in client_copy_list], factor, device=device
        )

        # Update client model weights
        for i in range(num_clients):
            client_copy_list[i].model.load_state_dict(aggregated_client.state_dict())
        
        # Inference
        if test_loader is not None:
            test_correct = 0
            test_loss = []
            with torch.no_grad():
                for samples, labels in test_loader:
                    samples = samples.to(device).double() if use_64bit \
                        else samples.to(device).float()
                    labels = labels.to(device).long()
                    splitting_output = aggregated_client(samples)
                    output = server.model(splitting_output)
                    batch_loss = server.criterion(output, labels)
                    test_loss.append(batch_loss.item())
                    _, predicted = torch.max(output.data, 1)
                    test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                loss = sum(test_loss) / len(test_loss)
                acc =  100. * test_correct / len(test_loader.dataset)
                logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {acc:.2f}%')

    return client_copy_list, aggregated_client, server.model, test_loss, acc

# ------------------------------------------------------------------------------
def sl_multi_server(
    rounds, iters, client_copy_list, server, train_loader_list, test_loader,
    factor, client_lr, server_lr, use_64bit=False, device='cpu'
):
    '''Split Learning with one server model per client

    Implements split learning with a dedicated server model per client which is
    averaged once per round.  Training happens for `rounds` rounds, with `iters`
    local iterations per client.

    Params
    ------
    rounds      - # training rounds
    iters       - # local iterations on each client, per round
    client_copy_list - list of Client() objects representing client model,
        optimizer and other options
    server      - Server() object representing the server model, optimizer and other
        options
    train_loader_list - List of DataLoader objects, one for each client's
        training set.
    test_loader - DataLoader object for test data
    factor      - Weights used in model aggregation
    client_lr, server_lr - Learning rates for client and server
    use_64bit   - [optional] Use 64 bit precision
    device      - [optional] CPU or GPU to use for computation

    Returns
    -------
    client_copy_list - list of all Client() objects after training
    aggregated_client - aggregated PyTorch client model
    aggregated_server - aggregated PyTorch server model
    test_loss        - Loss values on test data during training
    acc              - Accuracy values on test data during training

    '''
    logging.info(f"Configured rounds = {rounds}")

    num_clients = len(client_copy_list)
    server_copy_list = [copy.deepcopy(server) for _ in range(num_clients)]
    client_optim_ws = [optim.Adam(c.model.parameters(), lr=client_lr) for c in client_copy_list]
    server_optim_ws = [optim.Adam(s.model.parameters(), lr=server_lr) for s in server_copy_list]

    it_list = [iter(tl) for tl in train_loader_list]
    num_resets = [0 for _ in range(num_clients)]

    for r in range(rounds):
        for i in range(num_clients):
            for k in range(iters):

                # check if data iterator has finished iterating current cycle
                if (r * iters + k) == (num_resets[i] + 1) * len(it_list[i]):
                    num_resets[i] += 1
                    it_list[i] = iter(train_loader_list[i])

                # client feedforward
                samples, labels = next(it_list[i])
                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                client_optim_ws[i].zero_grad()
                server_optim_ws[i].zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)
                output = server_copy_list[i].model(splitting_output) 
                loss = server.criterion(output, labels)
                loss.backward()
                server_optim_ws[i].step()
                client_optim_ws[i].step()

        # aggregate client and server models
        aggregated_client = aggregate_models(
            [c.model for c in client_copy_list], factor, device=device
        )
        aggregated_server = aggregate_models(
            [s.model for s in server_copy_list], factor, device=device
        )

        # Update client and server weights
        for i in range(num_clients):
            client_copy_list[i].model.load_state_dict(aggregated_client.state_dict())
        
        for i in range(num_clients):
            server_copy_list[i].model.load_state_dict(aggregated_server.state_dict())

        # Inference
        if test_loader is not None:
            test_correct = 0
            test_loss = []
            with torch.no_grad():
                for samples, labels in test_loader:
                    samples = samples.to(device).double() if use_64bit \
                        else samples.to(device).float()
                    labels = labels.to(device).long()
                    splitting_output = aggregated_client(samples)
                    output = aggregated_server(splitting_output)
                    batch_loss = server.criterion(output, labels)
                    test_loss.append(batch_loss.item())
                    _, predicted = torch.max(output.data, 1)
                    test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
                loss = sum(test_loss) / len(test_loss)
                acc =  100. * test_correct / len(test_loader.dataset)
                logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {acc:.2f}%')

    return client_copy_list, aggregated_client, aggregated_server, test_loss, acc

# ------------------------------------------------------------------------------