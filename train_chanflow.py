import models.channel_flow as chan
import numpy as np
import torch
import os
import argparse
import time

if __name__ == '__main__':
    # check for CUDA
    parser = argparse.ArgumentParser(description='ChannelFlow command-line args')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disables CUDA')
    parser.add_argument('--disable-status', action='store_true',
                        help='Disables tqdm status bar')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # get hyperparams!
    hypers = chan.get_hyperparams(
                  ymin=-1,
                  ymax=1,
                  num_epochs=200000,
                  lr=0.0001,
                  num_layers=4,
                  num_units=40,
                  batch_size=1000,
                  weight_decay=.1
            )

    delta = (hypers['ymax']-hypers['ymin'])/2
    reynolds_stress = chan.get_mixing_len_model(hypers['k'], delta, hypers['dp_dx'], hypers['rho'], hypers['nu'])

    # set nu for to control Retau (default is nu=0.0001 which => Retau=1000)
    hypers['nu']=0.005555555555

    # calculate and print Retau for confirmation
    retau=chan.calc_retau(delta, hypers['dp_dx'], hypers['rho'], hypers['nu'])
    print('Training at Retau={}'.format(retau))

    # initialize Net and run training
    pdenn = chan.Chanflow(num_units=hypers['num_units'], num_layers=hypers['num_layers']).to(device=args.device)
    losses, pdenn_best = pdenn.train(hypers['ymin'], hypers['ymax'],
                                   reynolds_stress,
                                   nu=hypers['nu'],
                                   dp_dx=hypers['dp_dx'],
                                   rho=hypers['rho'],
                                   batch_size=hypers['batch_size'],
                                   epochs=hypers['num_epochs'],
                                   lr=hypers['lr'],
                                   weight_decay=hypers['weight_decay'],
                                   device=args.device,
                                   disable_status=args.disable_status)

    # save preds, losses, hyperparams, and model in a timestamped directory
    y=np.linspace(-1,1,1000)
    preds = pdenn_best.predict(torch.tensor(y.reshape(-1,1), dtype=torch.float)).detach().numpy()
    timestamp=time.time()
    retau=np.round(retau, decimals=2)
    os.mkdir('data/{}'.format(timestamp))
    np.save('data/{}/mixlen_preds_u{}.npy'.format(timestamp, retau), preds)
    np.save('data/{}/mixlen_loss_u{}.npy'.format(timestamp, retau), np.array(losses))
    np.save('data/{}/mixlen_hypers_u{}.npy'.format(timestamp, retau), hypers)
    torch.save(pdenn_best.state_dict(), 'data/{}/mixlen_model_u{}.pt'.format(timestamp, retau))
