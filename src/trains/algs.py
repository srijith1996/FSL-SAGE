# ------------------------------------------------------------------------------
import logging
import copy
import torch
from torch import optim
from utils.utils import calculate_load

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

    # Won't add to the comm load here because I'll add it for when algs load
    # in the agg model
    return aggregated

# ------------------------------------------------------------------------------
def fed_avg(
    rounds, model, criterion, train_loader_list, test_loader, factor,
    lr, use_64bit=False, device='cpu'
):
    '''Federated Averaging

    Implements the FedAvg algorithm given a model.  Training happens for
    `rounds` rounds, with one epoch worth of local iterations per client.

    Params
    ------
    rounds      - # training rounds
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

    model_optim = [optim.Adam(c.parameters(), lr=lr) for c in clients]

    all_loss = []
    all_acc = []
    # Store comm_load
    comm_load_list = []
    comm_load = 0
    for r in range(rounds):
        for i in range(num_clients):
            for samples, labels in train_loader_list[i]:

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

            # 2x because upload and download copies to server
            comm_load += 2 * calculate_load(clients[i])
            
        comm_load_list.append(comm_load)

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
            all_loss.append(loss)
            acc =  test_correct / len(test_loader.dataset)
            all_acc.append(acc)
            logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {100. * acc:.2f}%')

    return aggregated_client, all_loss, all_acc, comm_load_list

# ------------------------------------------------------------------------------
def sl_single_server(
    rounds, client_copy_list, server, train_loader_list, test_loader,
    factor, client_lr, server_lr, use_64bit=False, device='cpu'
):
    '''Split Learning with single server model

    Implements split learning given a list of client models and a server model.
    Training happens for `rounds` rounds, with one epoch worth of local
    iterations per client.

    Params
    ------
    rounds      - # training rounds
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

    all_loss = []
    all_acc = []
    comm_load_list = []
    comm_load = 0
    for r in range(rounds):
        for i in range(num_clients):
            for samples, labels in train_loader_list[i]:

                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                client_optim_ws[i].zero_grad()
                server_optim_ws.zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)

                # Represents the uploaded data
                smashed_data = splitting_output.clone().detach().requires_grad_(True)

                # Comm cost for upload splitting output to server
                comm_load += smashed_data.numel() * smashed_data.element_size() 

                output = server.model(smashed_data) 
                loss = server.criterion(output, labels)
                loss.backward()

                # Comm cost for downloading grads of smashed data
                comm_load += smashed_data.grad.numel() * smashed_data.grad.element_size()

                # Backprop split output with smashed data grad
                splitting_output.backward(smashed_data.grad)

                server_optim_ws.step()
                client_optim_ws[i].step()

        # aggregate client model
        aggregated_client = aggregate_models(
            [c.model for c in client_copy_list], factor, device=device
        )

        # Update client model weights
        for i in range(num_clients):
            client_copy_list[i].model.load_state_dict(aggregated_client.state_dict())
            
            # 2x because upload and download copies to server
            comm_load += 2 * calculate_load(client_copy_list[i].model)

        comm_load_list.append(comm_load)

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
                all_loss.append(loss)
                acc =  test_correct / len(test_loader.dataset)
                all_acc.append(acc)
                logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {100. * acc:.2f}%')

    return client_copy_list, aggregated_client, server.model, all_loss, all_acc, comm_load_list

# ------------------------------------------------------------------------------
def sl_multi_server(
    rounds, client_copy_list, server, train_loader_list, test_loader,
    factor, client_lr, server_lr, use_64bit=False, device='cpu'
):
    '''Split Learning with one server model per client

    Implements split learning with a dedicated server model per client which is
    averaged once per round.  Training happens for `rounds` rounds, with one
    epoch worth of local iterations per client.

    Params
    ------
    rounds      - # training rounds
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

    all_loss = []
    all_acc = []
    comm_load_list = []
    comm_load = 0
    for r in range(rounds):
        for i in range(num_clients):
            for samples, labels in train_loader_list[i]:

                samples = samples.to(device).double() if use_64bit \
                    else samples.to(device).float()
                labels = labels.to(device).long()
                client_optim_ws[i].zero_grad()
                server_optim_ws[i].zero_grad()

                # pass smashed data through full model 
                splitting_output = client_copy_list[i].model(samples)

                # Represents the uploaded data
                smashed_data = splitting_output.clone().detach().requires_grad_(True)

                # Upload the smashed data to the server
                comm_load += smashed_data.numel() * smashed_data.element_size() 

                output = server_copy_list[i].model(smashed_data) 
                loss = server.criterion(output, labels)
                loss.backward()

                # Download gradients of the smashed data
                comm_load += smashed_data.grad.numel() * smashed_data.grad.element_size() 

                # Backprop grads back to splitting_output
                splitting_output.backward(smashed_data.grad)

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
            comm_load += 2 * calculate_load(client_copy_list[i].model)
        
        for i in range(num_clients):
            server_copy_list[i].model.load_state_dict(aggregated_server.state_dict())

        comm_load_list.append(comm_load)

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
                all_loss.append(loss)
                acc =  test_correct / len(test_loader.dataset)
                all_acc.append(acc)
                logging.info(f' > Round {r}, testing loss: {loss:.2f}, testing acc: {100. * acc:.2f}%')

    return client_copy_list, aggregated_client, aggregated_server,\
        all_loss, all_acc, comm_load_list

# ------------------------------------------------------------------------------