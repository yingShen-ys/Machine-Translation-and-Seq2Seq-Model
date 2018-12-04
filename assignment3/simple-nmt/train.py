import argparse
import sys

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer
import simple_nmt.rl_trainer as rl_trainer
import simple_nmt.minimum_risk_trainer as minimum_risk_trainer
from simple_nmt.BIMPM import BIMPM
import pickle

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model',
                   required=True,
                   help='Model file name to save. Additional information would be annotated to the file name.'
                   )
    p.add_argument('--train',
                   required=True,
                   help='Training set file name except the extention. (ex: train.en --> train)'
                   )
    p.add_argument('--valid',
                   required=True,
                   help='Validation set file name except the extention. (ex: valid.en --> valid)'
                   )
    p.add_argument('--lang',
                   required=True,
                   help='Set of extention represents language pair. (ex: en + ko --> enko)'
                   )
    p.add_argument('--exts',
                   default='premises,hypothesis,is_premise,label',
                   help='all the other extensions'
                   )
    p.add_argument('--bimpm_pretrained_model_path',
                   default='BIMPM_best.pt',
                   help='BIMPM best model'
                   )
    p.add_argument('--snli_label_path',
                   default='label_vocab',
                   help='BIMPM label vocab'
                   )
    p.add_argument('--gpu_id',
                   type=int,
                   default=-1,
                   help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=-1'
                   )

    p.add_argument('--batch_size',
                   type=int,
                   default=64,
                   help='Mini batch size for gradient descent. Default=32'
                   )
    p.add_argument('--n_epochs',
                   type=int,
                   default=13,
                   help='Number of epochs to train. Default=13'
                   )
    p.add_argument('--print_every',
                   type=int,
                   default=1000,
                   help='Number of gradient descent steps to skip printing the training status. Default=1000'
                   )
    p.add_argument('--early_stop',
                   type=int,
                   default=-1,
                   help='The training will be stopped if there is no improvement this number of epochs. Default=-1'
                   )

    p.add_argument('--max_length',
                   type=int,
                   default=80,
                   help='Maximum length of the training sequence. Default=80'
                   )
    p.add_argument('--dropout',
                   type=float,
                   default=.2,
                   help='Dropout rate. Default=0.2'
                   )
    p.add_argument('--word_vec_dim',
                   type=int,
                   default=256,
                   help='Word embedding vector dimension. Default=256'
                   )
    p.add_argument('--hidden_size',
                   type=int,
                   default=512,
                   help='Hidden size of LSTM. Default=512'
                   )
    p.add_argument('--n_layers',
                   type=int,
                   default=2,
                   help='Number of layers in LSTM. Default=2'
                   )

    p.add_argument('--max_grad_norm',
                   type=float,
                   default=5.,
                   help='Threshold for gradient clipping. Default=5.0'
                   )
    p.add_argument('--adam',
                   action='store_true',
                   help='Use Adam instead of using SGD.'
                   )
    p.add_argument('--lr',
                   type=float,
                   default=1.,
                   help='Initial learning rate. Default=1.0'
                   )
    p.add_argument('--min_lr',
                   type=float,
                   default=.000001,
                   help='Minimum learning rate. Default=.000001'
                   )
    p.add_argument('--lr_decay_start_at',
                   type=int,
                   default=10,
                   help='Start learning rate decay from this epoch.'
                   )
    p.add_argument('--lr_slow_decay',
                   action='store_true',
                   help='Decay learning rate only if there is no improvement on last epoch.'
                   )
    p.add_argument('--lr_decay_rate',
                   type=float,
                   default=.5,
                   help='Learning rate decay rate. Default=0.5'
                   )

    p.add_argument('--rl_lr',
                   type=float,
                   default=.01,
                   help='Learning rate for reinforcement learning. Default=.01'
                   )
    p.add_argument('--n_samples',
                   type=int,
                   default=1,
                   help='Number of samples to get baseline. Default=1'
                   )
    p.add_argument('--rl_n_epochs',
                   type=int,
                   default=10,
                   help='Number of epochs for reinforcement learning. Default=10'
                   )
    p.add_argument('--rl_n_gram',
                   type=int,
                   default=4,
                   help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=4'
                   )
    p.add_argument('--temperature',
                   type=float,
                   default=1.0,
                   help='Temprature for minimum risk learning. Default=1.0'
                   )
    p.add_argument('--use_minimum_risk',
                   action='store_true',
                   help='Use minimum risk training instead of using REINFORCE.'
                   )
    p.add_argument('--pretrain',
                   action='store_true',
                   help='Specify whether it is pretrain or actual training.'
                   )
    p.add_argument('--tgt_vocab_path',
                   help='Specify the target vocab path.'
                   )
    p.add_argument('--reward_mode',
                   help='Specify the reward mode.'
                   )

    config = p.parse_args()

    return config


def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for key in vars(prev_config).keys():
        if key == 'pretrain':
            continue

        if '--%s' % key not in sys.argv or key == '--model':
            if vars(config).get(key) is not None:
                vars(config)[key] = vars(prev_config)[key]
            else:
                # Missing argument
                print('WARNING!!! Argument "-%s" is not found in current argument parser.\tSaved value:' % key, vars(prev_config)[key])
        else:
            # Argument value is change from saved model.
            print('WARNING!!! Argument "-%s" is not loaded from saved model.\tPrevious value and Current value:' % key, vars(prev_config)[key], vars(config)[key])

    return config


if __name__ == "__main__":
    config = define_argparser()

    import os.path
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.model):
        if config.gpu_id >= 0:
            saved_data = torch.load(config.model)
        else:
            saved_data = torch.load(config.model, map_location='cpu')

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)
        config.lr = saved_data['current_lr']
    else:
        saved_data = None

    # Load training and validation data set.
    print(config)
    loader = DataLoader(config.train,
                        config.valid,
                        [config.lang[:2], config.lang[-2:]] + config.exts.split(','),
                        batch_size=config.batch_size,
                        device=config.gpu_id,
                        max_length=config.max_length,
                        is_pretrain=config.pretrain
                        )

    loader.load_label_vocab(pickle.load(open(config.snli_label_path, 'rb')))
    if saved_data:
        loader.load_vocab(saved_data['src_vocab'], saved_data['tgt_vocab'])
    if config.tgt_vocab_path:
        loader.load_target_vocab(pickle.load(open(config.tgt_vocab_path, 'rb')))

    # Encoder's embedding layer input size
    input_size = len(loader.src.vocab)
    # Decoder's embedding layer input size and Generator's softmax layer output size
    output_size = len(loader.tgt.vocab)
    # Declare the model
    model = Seq2Seq(input_size,
                    config.word_vec_dim,  # Word embedding vector size
                    config.hidden_size,  # LSTM's hidden vector size
                    output_size,
                    n_layers=config.n_layers,  # number of layers in LSTM
                    dropout_p=config.dropout  # dropout-rate in LSTM
                    )

    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)
    loss_weight[data_loader.PAD] = 0.
    # Instead of using Cross-Entropy loss, we can use Negative Log-Likelihood(NLL) loss with log-probability.
    criterion = nn.NLLLoss(weight=loss_weight, size_average=False)

    assert config.reward_mode in ['nli', 'bleu', 'combined'], "the reward mode should be one of ['nli', 'bleu', 'combined']"

    # Pass models to GPU device if it is necessary.
    print(config.gpu_id)
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        criterion.cuda(config.gpu_id)

    # If we have loaded model weight parameters, use that weights for declared model.
    if saved_data is not None:
        model.load_state_dict(saved_data['model'])

    # Start training. This function maybe equivalant to 'fit' function in Keras.
    if config.n_epochs > 0:
        trainer.train_epoch(model,
                            criterion,
                            loader.train_iter,
                            loader.valid_iter,
                            config,
                            start_epoch=saved_data['epoch'] if saved_data is not None else 1,
                            others_to_save={'src_vocab': loader.src.vocab,
                                            'tgt_vocab': loader.tgt.vocab
                                            }  # We can put any object here to save with model.
                            )

    # Start reinforcement learning.
    if config.rl_n_epochs > 0:
        if config.gpu_id >= 0:
            saved_bimpm_model = torch.load(config.bimpm_pretrained_model_path)
        else:
            saved_bimpm_model = torch.load(config.bimpm_pretrained_model_path, map_location='cpu')

        bimpm = BIMPM(saved_bimpm_model['args'])
        bimpm.load_state_dict(saved_bimpm_model['model'])
        if config.gpu_id >= 0:
            bimpm.cuda()
        bimpm.training = False

        if config.use_minimum_risk:
            trainer = minimum_risk_trainer
        else:
            trainer = rl_trainer

        trainer.train_epoch(model,
                            bimpm,
                            criterion,  # Although it does not use cross-entropy loss, but its equation equals to use entropy.
                            loader.train_iter,
                            loader.valid_iter,
                            config,
                            #    start_epoch=(saved_data['epoch'] - config.n_epochs) if saved_data is not None else 1,
                            start_epoch=1,
                            others_to_save={'src_vocab': loader.src.vocab,
                                            'tgt_vocab': loader.tgt.vocab
                                            }
                            )
