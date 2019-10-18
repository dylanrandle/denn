# testing  sampling methods
import channel_flow as chan
import utils
torch.random.manual_seed(123)

sampling = ['grid', 'perturb', 'boundary', 'uniform']
HYPERS={'num_epochs': 100000}
for s in sampling:
    print('Training with sampling : {}'.format(s))
    HYPERS['sampling'] = s
    pdenn = chan.Chanflow(**HYPERS)
    now_hypers = pdenn.hypers
    run_dict = pdenn.train(save_run=True, disable_status=True)
