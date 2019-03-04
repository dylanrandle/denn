## run cross-validation on kappa (really just run training at various kappas)
import channel_flow as chan
import utils
torch.random.manual_seed(123)

kappas = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
HYPERS={'num_epochs': 100000, 'sampling': 'grid'}
for k in kappas:
    print('Training with kappa={}'.format(k))
    HYPERS['k'] = k
    pdenn = chan.Chanflow(**HYPERS)
    now_hypers = pdenn.hypers
    run_dict = pdenn.train(save_run=True, disable_status=True)
