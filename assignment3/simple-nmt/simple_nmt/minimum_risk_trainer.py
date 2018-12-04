import time
import numpy as np
# from nltk.translate.bleu_score import sentence_bleu as score_func
from nltk.translate.gleu_score import sentence_gleu as score_func
# from utils import score_sentence as score_func

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils

import utils
import data_loader


def get_bleu_reward(y, y_hat, n_gram=6):
    # This method gets the reward based on the sampling result and reference sentence.
    # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
    # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.

    # Since we don't calculate reward score exactly as same as multi-bleu.perl,
    # (especialy we do have different tokenization,) I recommend to set n_gram to 6.

    # |y| = (batch_size, length1)
    # |y_hat| = (batch_size, length2)

    scores = []

    # Actually, below is really far from parallized operations.
    # Thus, it may cause slow training.
    for b in range(y.size(0)):
        ref = []
        hyp = []
        for t in range(y.size(1)):
            ref += [str(int(y[b, t]))]
            if y[b, t] == data_loader.EOS:
                break

        for t in range(y_hat.size(1)):
            hyp += [str(int(y_hat[b, t]))]
            if y_hat[b, t] == data_loader.EOS:
                break

        # for nltk.bleu & nltk.gleu
        scores += [score_func([ref], hyp, max_len=n_gram) * 100.]

        # for utils.score_sentence
        # scores += [score_func(ref, hyp, 4, smooth = 1)[-1] * 100.]
    scores = torch.FloatTensor(scores).to(y.device)
    # |scores| = (batch_size)

    return scores

def get_nli_reward(pred_label, label, criterion):
    return criterion(pred_label, label)

def get_accuracy(logit, labels):
    prob = nn.functional.softmax(logit, dim=1)
    pred_labels = torch.argmax(prob, dim=1).long()

    return (pred_labels.long() == labels.long()).sum().item()/labels.size(0)

def padding_three_tensors(indice, premise, hypothesis, batch_size):
    pred_length = indice.size(1)
    hypothesis_length = hypothesis.size(1)
    premise_length = premise.size(1)

    # pad premise and hypothesis first, if needed
    max_length = max(pred_length, hypothesis_length, premise_length)
    if max_length != pred_length:
        padding_length = max_length - pred_length
        padding_tensor = torch.zeros(batch_size, padding_length).long()
        if indice.is_cuda:
            padding_tensor = padding_tensor.cuda()
        indice = torch.cat([indice, padding_tensor], 1)
        
    if max_length != hypothesis_length:
        padding_length = max_length - hypothesis_length
        padding_tensor = torch.zeros(batch_size, padding_length).long()
        if hypothesis.is_cuda:
            padding_tensor = padding_tensor.cuda()
        hypothesis = torch.cat([hypothesis, padding_tensor], 1)
        
    if max_length != premise_length:
        padding_length = max_length - premise_length
        padding_tensor = torch.zeros(batch_size, padding_length).long()
        if premise.is_cuda:
            padding_tensor = padding_tensor.cuda()
        premise = torch.cat([premise, padding_tensor], 1)

    return indice, premise, hypothesis

def pad_probs(probs, max_sample_length):
    result = []
    for prob in probs:
        sample_length = prob.size(1)
        batch_size = prob.size(0)
        if sample_length != max_sample_length:
            padding_length = max_sample_length - sample_length
            padding_tensor = torch.zeros(batch_size, padding_length)
            if prob.is_cuda:
                padding_tensor = padding_tensor.cuda()
            prob = torch.cat([prob, padding_tensor], 1)
        result.append(prob)
    
    return result

def train_epoch(model, bimpm, criterion, train_iter, valid_iter, config,
                start_epoch=1, others_to_save=None
                ):
    current_lr = config.rl_lr

    highest_valid_bleu = -np.inf
    no_improve_cnt = 0

    # Print initial valid BLEU before we start RL.
    model.eval()
    total_reward, sample_cnt = 0, 0
    for batch_index, batch in enumerate(valid_iter):
        current_batch_word_cnt = torch.sum(batch.tgt[1])
        x = batch.src
        y = batch.tgt[0][:, 1:]
        batch_size = y.size(0)
        # |x| = (batch_size, length)
        # |y| = (batch_size, length)

        # feed-forward
        y_hat, indice = model.search(x,
                                     is_greedy=True,
                                     max_length=config.max_length
                                     )
        # |y_hat| = (batch_size, length, output_size)
        # |indice| = (batch_size, length)

        reward = get_bleu_reward(y, indice, n_gram=config.rl_n_gram)

        total_reward += float(reward.sum())
        sample_cnt += batch_size
        if sample_cnt >= len(valid_iter.dataset.examples):
            break
    avg_bleu = total_reward / sample_cnt
    print("initial valid BLEU: %.4f" % avg_bleu)  # You can figure-out improvement.
    model.train()  # Now, begin training.

    # Start RL
    nli_criterion = nn.CrossEntropyLoss(reduce=False)
    print("start epoch:", start_epoch)
    print("number of epoch to complete:", config.rl_n_epochs + 1)

    parameters = model.parameters()  
    if config.reward_mode == 'combined':
        nli_weight = torch.rand(1)
        bleu_weight = torch.rand(1)
        nli_weight.requires_grad = True
        bleu_weight.requires_grad = True
        if config.gpu_id >= 0:
            nli_weight = nli_weight.cuda()
            bleu_weight = bleu_weight.cuda()
    
        print("nli_weight, bleu_weight:", nli_weight.data.cpu().numpy()[0], bleu_weight.data.cpu().numpy()[0])

        from itertools import chain
        parameters = chain(parameters, iter([nli_weight, bleu_weight]))

    # if config.adam:
    #     optimizer = optim.Adam(parameters, lr = current_lr)
    # else:
    optimizer = optim.SGD(parameters,
                        lr=current_lr,
                        momentum=0.9
                        )  # Default hyper-parameter is set for SGD.
    print("current learning rate: %f" % current_lr)
    print(optimizer)

    for epoch in range(start_epoch, config.rl_n_epochs + 1):
        sample_cnt = 0
        total_risk, total_errors, total_sample_count, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0, 0, 0
        start_time = time.time()
        train_loss = np.inf

        for batch_index, batch in enumerate(train_iter):
            optimizer.zero_grad()

            current_batch_word_cnt = torch.sum(batch.tgt[1])
            x = batch.src
            y = batch.tgt[0][:, 1:]
            batch_size = y.size(0)
            epoch_accuracy = []
            max_sample_length = 0
            sequence_probs, errors = [], []
            if config.reward_mode != 'bleu':
                premise = batch.premise
                hypothesis = batch.hypothesis
                isSrcPremise = batch.isSrcPremise
                label = batch.labels

            # |x| = (batch_size, length)
            # |y| = (batch_size, length)

            for _ in range(config.n_samples):
                # Take sampling process because set False for is_greedy.
                y_hat, indice = model.search(x, 
                                            is_greedy=False,
                                            max_length=config.max_length
                                            )
                max_sample_length = max(max_sample_length, indice.size(1))
                prob = y_hat.gather(2, indice.unsqueeze(2)).squeeze(2)
                sequence_probs.append(prob)
                # |prob| = (batch_size, length)

                if config.reward_mode == 'bleu':
                    bleu = get_bleu_reward(y, indice, n_gram=config.rl_n_gram)
                    reward = 100 - bleu
                    epoch_accuracy.append(bleu.sum()/batch_size)
                else:
                    padded_indice, premise, hypothesis = padding_three_tensors(indice, premise, hypothesis, batch_size)

                    # put pred sentece into either premise and hypothesis
                    for i in range(batch_size):
                        if isSrcPremise[i]:
                            premise[i] = padded_indice[i]
                        else:
                            hypothesis[i] = padded_indice[i]

                    kwargs = {'p': premise, 'h': hypothesis}
                    pred_logit = bimpm(**kwargs)
                    accuracy = get_accuracy(pred_logit, label)
                    epoch_accuracy.append(accuracy)

                # Based on the result of sampling, get reward.
                    if config.reward_mode == 'nli':
                        reward = -get_nli_reward(pred_logit, label, nli_criterion)
                    else:
                        reward = 1/(2 * nli_weight.pow(2)) * -get_nli_reward(pred_logit, label, nli_criterion) \
                            + 1/(2 * bleu_weight.pow(2)) * (100 - get_bleu_reward(y, indice, n_gram=config.rl_n_gram)) \
                            + torch.log(nli_weight * bleu_weight)
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)
                # |reward| = (batch_size)
                errors.append(reward)
            
            padded_probs = pad_probs(sequence_probs, max_sample_length)
            sequence_probs = torch.stack(padded_probs, dim = 2)
            # |sequence_probs| = (batch_size, max_sample_length, sample_size)
            errors = torch.stack(errors, dim = 1)
            # |errors| = (batch_size, sample_size)

            avg_probs = sequence_probs.sum(dim = 1)/max_sample_length
            if config.temperature != 1.0:
                probs = avg_probs.exp_().pow(1/config.temperature)
                probs = nn.functional.softmax(probs, dim=1)
            else:
                probs = nn.functional.softmax(avg_probs.exp_(), dim=1)
            risk = (probs * errors).sum()/batch_size
            risk.backward()

            # simple math to show stats
            total_risk += float(risk.sum())
            total_errors += float(reward.sum())
            total_sample_count += batch_size
            total_word_count += int(current_batch_word_cnt)
            total_parameter_norm += float(utils.get_parameter_norm(model.parameters()))
            total_grad_norm += float(utils.get_grad_norm(model.parameters()))

            if (batch_index + 1) % config.print_every == 0:
                avg_risk = total_risk / total_sample_count
                avg_errors = total_errors / total_sample_count
                avg_parameter_norm = total_parameter_norm / config.print_every
                avg_grad_norm = total_grad_norm / config.print_every
                avg_epoch_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
                elapsed_time = time.time() - start_time

                print("epoch: %d batch: %d/%d\t|param|: %.2f\t|g_param|: %.2f\trisk: %.4f\terror: %.4f\tAccuracy: %.2f\t%5d words/s %3d secs" % (epoch,
                                                                                                                               batch_index + 1,
                                                                                                                               int(len(train_iter.dataset.examples) // config.batch_size),
                                                                                                                               avg_parameter_norm,
                                                                                                                               avg_grad_norm,
                                                                                                                               avg_risk,
                                                                                                                               avg_errors,
                                                                                                                               avg_epoch_accuracy,
                                                                                                                               total_word_count // elapsed_time,
                                                                                                                               elapsed_time
                                                                                                                               ))
                
                if config.reward_mode == 'combined':
                    print("nli_weight, bleu_weight:", nli_weight.data.cpu().numpy()[0], bleu_weight.data.cpu().numpy()[0])

                total_risk, total_errors, total_sample_count, total_word_count, total_parameter_norm, total_grad_norm = 0, 0, 0, 0, 0, 0
                epoch_accuracy = []
                start_time = time.time()

                train_loss = avg_bleu

            # In orther to avoid gradient exploding, we apply gradient clipping.
            torch_utils.clip_grad_norm_(model.parameters(),
                                        config.max_grad_norm
                                        )
            # Take a step of gradient descent.
            optimizer.step()

            sample_cnt += batch_size
            if sample_cnt >= len(train_iter.dataset.examples):
                break

        sample_cnt = 0
        total_reward = 0

        # Start validation
        with torch.no_grad():
            model.eval()  # Turn-off drop-out

            for batch_index, batch in enumerate(valid_iter):
                current_batch_word_cnt = torch.sum(batch.tgt[1])
                x = batch.src
                y = batch.tgt[0][:, 1:]
                batch_size = y.size(0)
                # |x| = (batch_size, length)
                # |y| = (batch_size, length)

                # feed-forward
                y_hat, indice = model.search(x,
                                             is_greedy=True,
                                             max_length=config.max_length
                                             )
                # |y_hat| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                reward = get_bleu_reward(y, indice, n_gram=config.rl_n_gram)

                total_reward += float(reward.sum())
                sample_cnt += batch_size
                if sample_cnt >= len(valid_iter.dataset.examples):
                    break

            avg_bleu = total_reward / sample_cnt
            print("valid BLEU: %.4f" % avg_bleu)

            if highest_valid_bleu < avg_bleu:
                highest_valid_bleu = avg_bleu
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            model.train()

        model_fn = config.model.split(".")
        model_fn = model_fn[:-1] + ["%02d" % (config.n_epochs + epoch),
                                    "%.2f-%.4f" % (train_loss,
                                                   avg_bleu
                                                   )
                                    ] + [model_fn[-1]]

        # PyTorch provides efficient method for save and load model, which uses python pickle.
        to_save = {"model": model.state_dict(),
                   "config": config,
                   "epoch": config.n_epochs + epoch + 1,
                   "current_lr": current_lr
                   }
        if others_to_save is not None:
            for k, v in others_to_save.items():
                to_save[k] = v
        torch.save(to_save, '.'.join(model_fn))

        if config.early_stop > 0 and no_improve_cnt > config.early_stop:
            break
