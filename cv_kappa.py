## run cross-validation on kappa (really just run training at various kappas)
import channel_flow as chan
import utils

kappas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HYPERS={'num_epochs': 100000, 'sampling': 'grid'}
for k in kappas:
    print('Training with kappa={}'.format(k))
    HYPERS['k'] = k
    pdenn = chan.Chanflow(**HYPERS)
    now_hypers = pdenn.hypers
    run_dict = pdenn.train(save_run=True, disable_status=True)
