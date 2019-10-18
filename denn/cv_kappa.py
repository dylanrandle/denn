## run cross-validation on kappa (really just run training at various kappas)
import denn.channel_flow as chan
import utils
import torch
torch.random.manual_seed(123)

kappas = [0.38, 0.39, 0.40, 0.41, 0.42]
HYPERS={'num_epochs': 100000, 'sampling': 'perturb'}
for k in kappas:
    print('Training with kappa={}'.format(k))
    HYPERS['k'] = k
    pdenn = chan.Chanflow(**HYPERS)
    now_hypers = pdenn.hypers
    run_dict = pdenn.train(save_run=True, disable_status=True)
