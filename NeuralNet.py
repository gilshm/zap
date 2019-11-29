import torch
import time
import os
import Config as cfg
import numpy as np
from tabulate import tabulate
from StatsLogger import StatsLogger
from ThresholdOptimizer import ThresholdOptimizer


class NeuralNet:
    def __init__(self, arch, dataset, model_chkp=None, pretrained=True):
        """
        NeuralNet class wraps a model architecture and adds the functionality of training and testing both
        the entire model and the prediction layers.

        :param arch: a string that represents the model architecture, e.g., 'alexnet'
        :param dataset: a string that represents the dataset, e.g., 'cifar100'
        :param model_chkp: a model checkpoint path to be loaded (default: None)
        :param pretrained: whether to load PyTorch pretrained parameters, used for ImageNet (default: True)
        """
        cfg.LOG.write('__init__: arch={}, dataset={}, model_chkp={}, pretrained={}'
                      .format(arch, dataset, model_chkp, pretrained))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')

        self.arch = '{}_{}'.format(arch, dataset)

        # The 'pretrained' argument is used for PyTorch pretrained ImageNet models, whereas the model_chkp is
        # intended to load user checkpoints. At this point model_chkp and pretrained are not supported together,
        # although it is possible.
        if model_chkp is None:
            self.model = cfg.MODELS[self.arch](pretrained=pretrained)
        else:
            self.model = cfg.MODELS[self.arch]()

        # self.parallel_model = torch.nn.DataParallel(self.model).cuda(self.device)
        # Disabled parallel model. Statistics collection does not support GPU parallelism.
        self.parallel_model = self.model.cuda(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.device)
        self.criterion_pred = torch.nn.MSELoss().cuda(self.device)

        self.optimizer = None
        self.lr_plan = None
        self.best_top1_acc = 0
        self.next_train_epoch = 0

        if model_chkp is not None:
            self._load_state(model_chkp)

        self.stats = StatsLogger()
        self.stats.add_tbl('main')
        self.stats.add_tbl('mispred_values_hist')
        self.stats.add_tbl('mask_values_hist')
        self.stats.add_tbl('err_to_th')
        self.stats.add_tbl('roc')

        self.th_opt = ThresholdOptimizer()

    def test(self, test_gen, stats=None, reset_output=True):
        batch_time = self.AverageMeter('Time', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        top5 = self.AverageMeter('Acc@5', ':6.2f')
        progress = self.ProgressMeter(len(test_gen), batch_time, losses, top1, top5, prefix='Test: ')

        # Switch to evaluate mode
        self.model.eval()
        self.model.set_train(False)
        self.model.reset_pred_stats()
        self.model.add_stats_gather(stats)

        if reset_output:
            self.stats.clear_tbl('main')
            self.stats.clear_tbl('mispred_values_hist')
            self.stats.clear_tbl('mask_values_hist')
            self.stats.clear_tbl('err_to_th')

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_gen):
                input = input.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)

                # Compute output
                output = self.parallel_model(input)
                loss = self.criterion(output, target)

                # Measure accuracy and record loss
                acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Print to screen
                if i % 100 == 0:
                    progress.print(i)

            # TODO: this should also be done with the ProgressMeter
            cfg.LOG.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        # Gather and print statistics
        if stats:
            self._print_stats(stats, reset_output)

        return top1.avg

    def train(self, train_gen, test_gen, epochs, lr=0.0001, lr_plan=None, momentum=0.9, wd=5e-4, stats=None):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        self.lr_plan = lr_plan

        cfg.LOG.write('train_pred: epochs={}, lr={}, lr_plan={}, momentum={}, wd={}, batch_size={}, optimizer={}'
                      .format(epochs, lr, lr_plan, momentum, wd, cfg.BATCH_SIZE, 'SGD'))

        for epoch in range(self.next_train_epoch, epochs):
            self._adjust_lr_rate(self.optimizer, epoch, lr_plan)
            self._train_step(train_gen, epoch, self.optimizer)
            torch.cuda.empty_cache()
            top1_acc = self.test(test_gen, stats=stats).item()

            if top1_acc > self.best_top1_acc:
                self.best_top1_acc = top1_acc
                self._save_state(epoch)

    def train_pred(self, train_gen, test_gen, epochs, pred_idx, lr=0.01, lr_plan=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        self.lr_plan = lr_plan

        cfg.LOG.write('train_pred: epochs={}, pred_idx={}, lr={}, lr_plan={}, batch_size={}, optimizer={}'
                      .format(epochs, pred_idx, lr, lr_plan, cfg.BATCH_SIZE, 'Adam'))

        for epoch in range(epochs):
            self._adjust_lr_rate(self.optimizer, epoch, lr_plan)
            self._train_step_pred(train_gen, epoch, self.optimizer, pred_idx)
            self.test(test_gen, stats=[cfg.STATS_GENERAL])

            pred_layer = self.model.pred_layers[pred_idx]
            total_activations = pred_layer.stats['X_o>0'] + pred_layer.stats['X_o<=0']
            total_saved = pred_layer.stats['M==0'] / total_activations

            filename = 'idx-{}_mask-{}_th-{}_saved-{}.pth'\
                .format(pred_idx, pred_layer.mask_type, pred_layer.threshold, round(100*total_saved, 2))
            pred_layer.save_state('{}/{}'.format(cfg.LOG.path, filename))
            cfg.LOG.write('', date=False)
            cfg.LOG.write('ZAP checkpoint saved to {}/{}'.format(cfg.LOG.path, filename))

    def _train_step(self, train_gen, epoch, optimizer):
        self.model.train()
        self.model.rm_all_stats_gather()

        batch_time = self.AverageMeter('Time', ':6.3f')
        data_time = self.AverageMeter('Data', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        top5 = self.AverageMeter('Acc@5', ':6.2f')
        progress = self.ProgressMeter(len(train_gen), batch_time, data_time, losses, top1,
                                      top5, prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (input, target) in enumerate(train_gen):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(self.device, non_blocking=True)
            target = target.cuda(self.device, non_blocking=True)

            # Compute output
            output = self.parallel_model(input)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.print(i)

    def _train_step_pred(self, train_gen, epoch, optimizer, pred_idx):
        batch_time = self.AverageMeter('Time', ':6.3f')
        data_time = self.AverageMeter('Data', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        progress = self.ProgressMeter(len(train_gen), batch_time, data_time, losses,
                                      prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (input, _) in enumerate(train_gen):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(self.device, non_blocking=True)

            pred_layer = self.model.pred_layers[pred_idx]

            # Disable all layers
            self.model.eval()
            self.model.set_train(False)
            self.model.disable_pred_layers()
            self.model.disable_grad()
            self.model.rm_all_stats_gather()

            # Only enable the pred layer training
            pred_layer.train()
            pred_layer.set_train(True)
            pred_layer.enable_layer()
            pred_layer.enable_grad()
            self.model(input)

            x_orig = pred_layer.x_orig
            x_pred = pred_layer.x_pred

            # The next two lines will add the pre-computed 1's to x_orig
            x_orig = torch.mul((x_orig > 0).float(),
                               pred_layer.mask_c[0:pred_layer.x_orig.shape[0], 0:pred_layer.x_orig.shape[1],
                                                 0:pred_layer.x_orig.shape[2], 0:pred_layer.x_orig.shape[3]])
            x_orig = torch.add(x_orig,
                               pred_layer.mask[0:pred_layer.x_orig.shape[0], 0:pred_layer.x_orig.shape[1],
                                               0:pred_layer.x_orig.shape[2], 0:pred_layer.x_orig.shape[3]])

            loss = self.criterion_pred(x_orig, x_pred)

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.print(i)

    def _adjust_lr_rate(self, optimizer, epoch, lr_dict):
        if lr_dict is None:
            return

        for key, value in lr_dict.items():
            if epoch == key:
                cfg.LOG.write("=> New learning rate set to {}".format(value))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = value

    def _save_state(self, epoch):
        filename = '{}_epoch-{}_top1-{}.pth'.format(self.arch, epoch, round(self.best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.path, filename)

        state = {'arch': self.arch,
                 'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'lr_plan': self.lr_plan,
                 'best_top1_acc': self.best_top1_acc}

        torch.save(state, path)

    def _load_state(self, path):
        if os.path.isfile(path):
            chkp = torch.load(path)

            # Load class variables from checkpoint
            assert (self.arch == chkp['arch'])
            self.next_train_epoch = chkp['epoch']
            try:
                self.model.load_state_dict(chkp['state_dict'], strict=False)
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, probably size mismatch due to prediction layers, this is ' +
                              'OK as long as the errors are only related to ZAP layers')
                cfg.LOG.write('{}'.format(e))

            self.optimizer_state = chkp['optimizer']
            self.lr_plan = chkp['lr_plan']
            self.best_top1_acc = chkp['best_top1_acc']
            cfg.LOG.write("Checkpoint best top1 accuracy is {} @ epoch {}"
                          .format(round(self.best_top1_acc, 2), self.next_train_epoch - 1))
        else:
            cfg.LOG.write("Unable to load model checkpoint from {}".format(path))
            raise RuntimeError

    def _print_stats(self, stats, reset_output):
        for pred_layer in self.model.pred_layers:
            if (not pred_layer.is_disabled() and not reset_output) or reset_output:
                if cfg.STATS_GENERAL in stats or cfg.STATS_ROC in stats:
                    values, headers = pred_layer.get_stats()
                    self.stats.add_headers('main', headers)
                    self.stats.add_row('main', values)

                if cfg.STATS_MISPRED_VAL_HIST in stats:
                    values, headers = pred_layer.get_mispred_values_hist()
                    self.stats.add_headers('mispred_values_hist', headers)
                    self.stats.add_row('mispred_values_hist', values)

                if cfg.STATS_MASK_VAL_HIST in stats:
                    values, headers = pred_layer.get_mask_values_hist()
                    self.stats.add_headers('mask_values_hist', headers)
                    self.stats.add_row('mask_values_hist', values)

                if cfg.STATS_ERR_TO_TH in stats:
                    values, headers = pred_layer.get_err_to_th()
                    self.stats.add_headers('err_to_th', headers)
                    self.stats.add_row('err_to_th', values)

        if cfg.STATS_GENERAL in stats:
            tbl = self.stats.get_tbl_normalized('main', ['X_o>0', 'X_o<=0'])
            cfg.LOG.write_title('ZAPs Statistics')
            cfg.LOG.write(tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

            tbl = self.stats.get_tbl_sum_normalized('main', ['main:X_o>0', 'main:X_o<=0'])
            cfg.LOG.write('\n' + tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

        if cfg.STATS_ROC in stats:
            # Data is based on 'main' table sum
            tbl = self.stats.get_tbl_sum('main',
                                         show_cols=['Miss-0->!0', 'Hit-!0->!0', 'Hit-0->0', 'Miss-!0->0'])
            self.stats.add_headers('roc', tbl['headers'])
            self.stats.add_row('roc', tbl['rows'][0])

            tbl = self.stats.get_tbl('roc')
            cfg.LOG.write_title('ROC', 'Needs post-processing in Excel')
            cfg.LOG.write(tabulate(tbl['rows'], headers=tbl['headers']), date=False)

        if cfg.STATS_MISPRED_VAL_HIST in stats:
            tbl = self.stats.get_tbl('mispred_values_hist')
            cfg.LOG.write_title('Misprediction Values Histogram')
            cfg.LOG.write(tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

            tbl = self.stats.get_tbl_sum_normalized('mispred_values_hist', ['main:X_o>0', 'main:X_o<=0'])
            cfg.LOG.write('\n' + tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

        if cfg.STATS_MASK_VAL_HIST in stats:
            tbl = self.stats.get_tbl('mask_values_hist')
            cfg.LOG.write_title('Mask Values Histogram')
            cfg.LOG.write(tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

            tbl = self.stats.get_tbl_sum_normalized('mask_values_hist', ['main:X_o>0', 'main:X_o<=0'])
            cfg.LOG.write('\n' + tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

        if cfg.STATS_ERR_TO_TH in stats:
            tbl = self.stats.get_tbl('err_to_th')
            cfg.LOG.write_title('Error = f(Threshold)')
            cfg.LOG.write(tabulate(tbl['rows'], headers=tbl['headers'], showindex="always"), date=False)

        cfg.LOG.write('', date=False)

    def load_state_pred(self, filename: str, pred_idx=None):
        filename_ = filename.split('/')[-1].split('_')

        if pred_idx is None:
            for param in filename_:
                if 'idx' in param:
                    pred_idx = int(param.split('-')[1])

        self.model.pred_layers[pred_idx].load_state(filename)
        cfg.LOG.write('Loading ZAP #{} from {}'.format(pred_idx, filename))

    def summary(self, x_size, print_it=True):
        return self.model.summary(x_size, print_it=print_it)

    def print_weights(self):
        self.model.print_weights()

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    class AverageMeter(object):
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

    class ProgressMeter(object):
        def __init__(self, num_batches, *meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def print(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            cfg.LOG.write('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'
