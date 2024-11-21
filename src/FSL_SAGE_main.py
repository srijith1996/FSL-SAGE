# -----------------------------------------------------------------------------
"""
Add local loss, no need to transmit the gradient
client transmit smashed data not every batch data
server part: model 0 batch 0 - model 0 batch 4 - model 1 batch 0 - model 1 batch 4  ...
"""

# -----------------------------------------------------------------------------
import os
import time
import math
import copy
import json
import random
import numpy as np
import torch
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from utils import options, utils, logs, plots
from trains import client, aux_models, algs
from trains import server as serv
import logging

# -----------------------------------------------------------------------------
# TODO: Move these to command line args
DEBUG = True
USE_64BIT = False
WARM_START = False
WARM_START_EPOCHS = 1
AGGREGATE_AUXILIARY_MODELS = False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if USE_64BIT: torch.set_default_dtype(torch.float64)
if DEBUG: torch.set_printoptions(sci_mode=True)

# -----------------------------------------------------------------------------
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)   

# -----------------------------------------------------------------------------
def calculate_load(model):        
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2    # MB
#     size_all_mb = (param_size + buffer_size)   # B
    return size_all_mb

# -----------------------------------------------------------------------------
def main(u_args, s_args, c_args):
    assert s_args["activated"] <= s_args["client"], \
        f"# activated clients {s_args['activated']} is greater than # clients {s_args['client']}"

    ## process dataset
    trainSet, testSet = utils.get_dataset(s_args, u_args) 
    client_train_set, client_test_set = utils.depart_dataset(u_args, s_args, trainSet, testSet)

    trainLoader_list = []
    for i in range(s_args["activated"]):
        train_set = client_train_set[i]["idxs"]
        trainLoader_list.append(
            DataLoader(
                utils.DatasetSplit(trainSet, train_set),
                batch_size=c_args['batch_size'],
                shuffle=True, pin_memory=False
            )
        )
    
    testLoader = DataLoader(
        testSet, batch_size=c_args['batch_size'], shuffle=False, pin_memory=False
    )
    
    # Define the server, and the list of client copies
    server = serv.Server(serv.Server_model_cifar(), s_args, device=DEVICE)
    client_copy_list = []
    
    for i in range(s_args["activated"]):   
        client_copy_list.append(client.Client(
            i, trainLoader_list[i],
            client.Client_model_cifar(),
            #aux_models.LinearAuxiliaryModel(2305, server, device=DEVICE, bias=True),
            #aux_models.LinearGradScalarAuxiliaryModel(
            #    2304, 10, server, device=DEVICE, align_epochs=100, align_step=5e-2
            #),
            aux_models.NNGradScalarAuxiliaryModel(
                2304, 10, server, device=DEVICE, n_hidden=2304,
                align_epochs=100, align_step=1e-3, align_batch_size=1000,
                max_dataset_size=1000
            ),
            c_args, device=DEVICE
        ))
    
    # Initial client & server model
    init_all(client_copy_list[0].model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0., std=0.05) 
    init_all(server.model, torch.nn.init.normal_, mean=0., std=0.05) 
    #init_all(client_copy_list[0].model, torch.nn.init.kaiming_normal_) 
    #init_all(client_copy_list[0].auxiliary_model, torch.nn.init.normal_, mean=0., std=1.0) 
    #init_all(server.model, torch.nn.init.kaiming_normal_) 
    
        
    for i in range(s_args["activated"]):
        client_copy_list[i].model.load_state_dict(client_copy_list[0].model.state_dict())
        client_copy_list[i].model.to(DEVICE)
        client_copy_list[i].auxiliary_model.load_state_dict(client_copy_list[0].auxiliary_model.state_dict())
        client_copy_list[i].auxiliary_model.to(DEVICE)
    server.model.to(DEVICE)    
    
        
    # # Calculate the weights for dataset size
    dataset_size_list = [client_copy_list[i].dataset_size for i in range(s_args["activated"])]
    total = sum(dataset_size_list)
    factor = [i / total for i in dataset_size_list]
    logging.info(f"Aggregation Factor: {factor}")


    r = 0  # current communication round
    l = c_args['align'] # the number of training steps to take before the alignment step 
    logging.info(f"Random seed: {str(seed)}")
    logging.info(f"Alignment interval (l): {str(l)}")

    # metrics to save
    acc_list = []
    loss_list = []
    tr_acc_list = []
    tr_loss_list = []
    comm_load_list = []
    if DEBUG:
        tr_loss_per_client_iter = [[] for _ in range(s_args['activated'])]
        tr_acc_per_client_iter = [[] for _ in range(s_args['activated'])]
        tr_grad_mse_per_client_iter = [[] for _ in range(s_args['activated'])]
        tr_grad_nmse_per_client_iter = [[] for _ in range(s_args['activated'])]

    start = time.time()
    comm_load = 0

    assert c_args['batch_size'] <= total  // s_args["activated"], \
        f"Chosen batch_size per client ({c_args['batch_size']}) is larger than the dataset size per client ({total // s_args['activated']})."

    # TODO: Right now the code assumes activated clients = total number of clients.
    # May need to change this later

    #it_list = [iter(tl) for tl in trainLoader_list]
    #num_resets = [0 for _ in range(s_args['activated'])]

    # WARM START
    if WARM_START:
        logging.info("----------------------------- WARM START USING SL -----------------------------------")
        client_copy_list, aggregated_client, server_model, tloss, tacc = algs.sl_single_server(
            WARM_START_EPOCHS, len(trainLoader_list[0]), client_copy_list, server,
            trainLoader_list, testLoader, factor, 1e-3, 1e-3, use_64bit=USE_64BIT,
            device=DEVICE
        )
        logging.info(f"After warm start: Test loss: {tloss[-1]:.2f}, Test accuracy: {tacc:.2f}")
        logging.info("-------------------------------------------------------------------------------------")

        # reload the server model and optimizer
        server = serv.Server(serv.Server_model_cifar(), s_args, device=DEVICE)
        server.model.load_state_dict(server_model.state_dict())
        server.model.to(DEVICE)

    set_mark = False
    dbg_saved_aux_params = [[] for _ in range(s_args['activated'])]
    for r in range(s_args["round"]):
        for i in range(s_args["activated"]):
            #for k in range(u_args["batch_round"]):
            for k, (samples, labels) in enumerate(trainLoader_list[i]):

                # check if data iterator has finished iterating current cycle
                #if (r * u_args["batch_round"] + k) == (num_resets[i] + 1) * len(it_list[i]):
                #    num_resets[i] += 1
                #    it_list[i] = iter(trainLoader_list[i])

                #samples, labels = next(it_list[i])
                samples = samples.to(DEVICE).double() if USE_64BIT else \
                    samples.to(DEVICE).float()
                labels = labels.to(DEVICE).long()
                
                # client feedforward
                client_copy_list[i].optimizer.zero_grad()
                splitting_output = client_copy_list[i].model(samples)
                local_smashed_data = splitting_output.clone().detach().requires_grad_(True)

                # contact server every p steps
                if k % u_args['batch_round'] == 0:

                    #logging.debug(f"R{r:2d} C{i:2d} B{k:2d} Sending smashed data to server")
                    smashed_data = splitting_output.clone().detach().requires_grad_(True)
                    #comm_load += smashed_data.numel() * 4   # float32 = 4 bytes

                    # pass smashed data through server
                    server.optimizer.zero_grad()
                    output = server.model(smashed_data) 
                    loss = server.criterion(output, labels)
                    loss.backward()
                    server.optimizer.step()

                    # save smashed data to memory
                    client_copy_list[i].auxiliary_model.add_datapoint(
                        splitting_output.clone().detach(), labels
                    )

                # perform alignment at every l^th round and first local iteration
                if r % l == 0 and k == 0:

                    if DEBUG: logging.debug(f" ------------ ALIGNMENT <R {r:2d}, C {i:2d}, k {k:2d}> -----------------")
                    # recompute gradients of smashed data
                    client_copy_list[i].auxiliary_model.refresh_data()

                    # Debug the newly computed grad approximation
                    if DEBUG:
                        set_mark = True
                        client_copy_list[i].auxiliary_model.debug_grad_nmse(
                            local_smashed_data, labels,
                            pre=f' -- [before align] <round {r:2d}, client {i:2d}, batch index {k:2d}>'
                        )
               
                    # perform alignment for current client
                    client_copy_list[i].auxiliary_model.align()

                    # debugging if server remains fixed until next update
                    dbg_saved_server_params = [p.clone().detach() for p in server.model.parameters()]
                    dbg_saved_aux_params[i] = [p.clone().detach() for p in client_copy_list[i].auxiliary_model.parameters()]

                # client backpropagation and update client-side model weights
                #client_copy_list[i].auxiliary_optimizer.zero_grad()
                client_grad_approx = client_copy_list[i].auxiliary_model(local_smashed_data, labels)

                # Debug the newly computed grad approximation
                if set_mark and r % l == 0: 
                    set_mark = False
                    client_copy_list[i].auxiliary_model.debug_grad_nmse(
                        local_smashed_data, labels,
                        pre=f' -- [after align] <round {r:2d}, client {i:2d}, batch index {k:2d}>'
                    )
                    if DEBUG:  logging.debug(f" -----------------------------------------------------------------------")
                
                # for debugging ********
                if False and DEBUG:
                    def assert_server_model_identical():
                        for p1, p2 in zip(dbg_saved_server_params, server.model.parameters()):
                            assert torch.allclose(p1, p2), "Server model unintentionally changed!!"
                    def assert_aux_models_identical():
                        for p1, p2 in zip(dbg_saved_aux_params[i], client_copy_list[i].auxiliary_model.parameters()):
                            assert torch.allclose(p1, p2), f"Auxiliary model for client {i} unintentionally changed!!"
                    assert_server_model_identical()
                    if not AGGREGATE_AUXILIARY_MODELS: assert_aux_models_identical()
                    mse, nmse = client_copy_list[i].auxiliary_model.debug_grad_nmse(
                        local_smashed_data, labels,
                        pre=f' -- [round {r:2d}, client {i:2d}, batch index {k:2d}]'
                    )
                    tr_grad_mse_per_client_iter[i].append(mse.item())
                    tr_grad_nmse_per_client_iter[i].append(nmse.item())

                    # Test on training data
                    with torch.no_grad():
                        tr_loss = []
                        tr_correct = 0

                        for samples, labels in trainLoader_list[i]:
                            samples = samples.to(DEVICE).double() if USE_64BIT else samples.to(DEVICE).float()
                            labels = labels.to(DEVICE).long()
                            output = server.model(aggregated_client(samples))
                            batch_loss = server.criterion(output, labels)
                            tr_loss.append(batch_loss.item())
                            _, predicted = torch.max(output.data, 1)
                            tr_correct += predicted.eq(labels.view_as(predicted)).sum().item()

                        tr_loss = sum(tr_loss) / len(tr_loss)
                        total = len(trainLoader_list[i].dataset)
                        tr_acc = tr_correct / total
                        logging.debug(f' tr. loss: {tr_loss:.2f}, tr. acc: {100. * tr_acc:.2f}%')

                        tr_loss_per_client_iter[i].append(tr_loss)
                        tr_acc_per_client_iter[i].append(tr_acc)

                splitting_output.backward(client_grad_approx)
                client_copy_list[i].optimizer.step()


        # Model Aggregation (weighted)
        aggregated_client = copy.deepcopy(client_copy_list[0].model)
        aggregated_client_weights = aggregated_client.state_dict()

        for key in aggregated_client_weights:
            aggregated_client_weights[key] = client_copy_list[0].model.state_dict()[key] * factor[0]

        if AGGREGATE_AUXILIARY_MODELS:
            aggregated_client_auxiliary = copy.deepcopy(client_copy_list[0].auxiliary_model)
            aggregated_client_weights_auxiliary = aggregated_client_auxiliary.state_dict()

            for key in aggregated_client_weights_auxiliary:
                aggregated_client_weights_auxiliary[key] = client_copy_list[0].auxiliary_model.state_dict()[key] * factor[0]

        for i in range(1, s_args["activated"]):
            for key in aggregated_client_weights:
                aggregated_client_weights[key] += client_copy_list[i].model.state_dict()[key] * factor[i]
            if AGGREGATE_AUXILIARY_MODELS:
                for key in aggregated_client_weights_auxiliary:
                    aggregated_client_weights_auxiliary[key] += client_copy_list[i].auxiliary_model.state_dict()[key] * factor[i]

        # Update client model weights and auxiliary weights
        for i in range(s_args["activated"]):
            client_copy_list[i].model.load_state_dict(aggregated_client_weights)
            comm_load += 2 * calculate_load(client_copy_list[i].model)
            if AGGREGATE_AUXILIARY_MODELS:
                client_copy_list[i].auxiliary_model.load_state_dict(aggregated_client_weights_auxiliary)
                comm_load += 2 * calculate_load(client_copy_list[i].auxiliary_model)

        # Inference
        aggregated_client.to(DEVICE)
        aggregated_client.load_state_dict(aggregated_client_weights)
        test_correct = 0
        test_loss = []
        with torch.no_grad():
            for samples, labels in testLoader:
                samples = samples.to(DEVICE).double() if USE_64BIT else samples.to(DEVICE).float()
                labels = labels.to(DEVICE).long()
                splitting_output = aggregated_client(samples)
                output = server.model(splitting_output)
                batch_loss = server.criterion(output, labels)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(output.data, 1)
                test_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            ts_loss = sum(test_loss) / len(test_loss)
            ts_acc = test_correct / len(testLoader.dataset)

        acc_list.append(ts_acc)
        loss_list.append(ts_loss)
        comm_load_list.append(comm_load)

        # Test on training data (debugging)
        with torch.no_grad():
            tr_loss = []
            tr_correct = 0
            for i in range(s_args['activated']):
                for samples, labels in trainLoader_list[i]:
                    samples = samples.to(DEVICE).double() if USE_64BIT else samples.to(DEVICE).float()
                    labels = labels.to(DEVICE).long()
                    output = server.model(aggregated_client(samples))
                    batch_loss = server.criterion(output, labels)
                    tr_loss.append(batch_loss.item())
                    _, predicted = torch.max(output.data, 1)
                    tr_correct += predicted.eq(labels.view_as(predicted)).sum().item()
            tr_loss = sum(tr_loss) / len(tr_loss)
            total = sum([len(trainLoader_list[i].dataset) for i in range(s_args['activated'])])
            tr_acc = tr_correct / total

        tr_acc_list.append(tr_acc)
        tr_loss_list.append(tr_loss)

        logging.info(f' > R {r:2d}, for the weighted aggregated final model, testing loss: {ts_loss:.4e}, testing acc: {100. * ts_acc:.2f}% ({test_correct:5d}/{len(testLoader.dataset)}), training loss: {tr_loss:.2f}, training acc: {100. * tr_acc:.2f}%')

    logging.info(f'The total running time for all rounds is {round(time.time() - start, 2)} seconds')

    # Save reults to .json files.
    results = {'test_loss': loss_list, 'test_acc' : acc_list,
               'comm_load' : comm_load_list, 'step': s_args['t_round']}

    if u_args['save']:
        file_name = os.path.join(u_args['save_path'], 'results.json')
        with open(file_name, 'w') as outf:
            json.dump(results, outf)
            logging.info(f"[NOTICE] Saved results to '{file_name}'.")
        
        metrics_file = os.path.join(u_args['save_path'], 'metrics.pt')
        torch.save([acc_list, loss_list, comm_load_list], metrics_file)

        # save trained model
        utils.save_model(aggregated_client, os.path.join(u_args['save_path'], 'agg_client.pt'))
        utils.save_model(server.model, os.path.join(u_args['save_path'], 'server.pt'))

    # Plot training and test results
    plots.plot_final_metrics(
        tr_loss_per_client_iter, tr_acc_per_client_iter,
        tr_grad_mse_per_client_iter, tr_grad_nmse_per_client_iter, tr_loss_list,
        tr_acc_list, loss_list, acc_list, u_args['plot_path'], debug=DEBUG
    )

    logging.info(f"Testing accuracy: {acc_list}")
    logging.info(f"Testing loss: {loss_list}")
    logging.info(f"Training accuracy: {tr_acc_list}")
    logging.info(f"Training loss: {tr_loss_list}")

# -----------------------------------------------------------------------------
if __name__ == '__main__':    
    ## get system configs
    args = options.args_parser('FSL-Approx')    #---------todo
    u_args, s_args, c_args = options.group_args(args) #---------todo
    if u_args['save']:
        u_args['plot_path'] = os.path.join(u_args['save_path'], 'plots')
        os.makedirs(u_args['plot_path'], exist_ok=True)
    else:
        u_args['plot_path'] = None
    utils.show_utils(u_args) #---------todo
    
    log_file = os.path.join(u_args['save_path'], "output.log") \
        if u_args['save'] else None
    logs.configure_logging(log_file)
    if u_args['save']: logs.log_hparams(u_args, c_args, s_args)

    logging.info(f"Using GPU: {torch.cuda.is_available()}")
 
    seed = u_args['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(u_args, s_args, c_args)

# -----------------------------------------------------------------------------