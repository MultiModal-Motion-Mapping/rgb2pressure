from tensorboardX import SummaryWriter
import torch
import numpy as np
import trimesh


class ContRecorder():
    def __init__(self, opt):
        self.iter = 0
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.save_freq = opt.record.save_freq
        self.show_freq = opt.record.show_freq
        self.print_freq = opt.record.print_freq

        self.checkpoint_path = opt.checkpoint_path
        self.result_path = self.logdir
        if opt.trainer.model == 'Vector':
            self.name = opt.name[0]
        else:
            self.name = opt.name[1]

    def init(self):
        self.iter = 0

    def logTensorBoard(self, l_data):
        self.logger.add_scalar('loss', l_data['loss'].item(), self.iter)
        self.logger.add_scalar('loss_press', l_data['loss_contact'].item(), self.iter)
        self.logger.add_scalar('loss_contact', l_data['loss_contact'].item(), self.iter)
        self.logger.add_scalar('loss_vec', l_data['loss_vec'].item(), self.iter)
        self.iter += 1

    def logPressNetTensorBoard(self, l_data):
        self.logger.add_scalar('loss', l_data['loss'].item(), self.iter)
        self.logger.add_scalar('loss_press', l_data['loss_press'].item(), self.iter)
        self.logger.add_scalar('loss_contact', l_data['loss_contact'].item(), self.iter)
        self.logger.add_scalar('loss_vec', l_data['loss_vec'].item(), self.iter)
        self.iter += 1

    def log(self, l_data):
        if (l_data['epoch']+1) % self.save_freq == 0:
            print('Save checkpoint to %s/%s/latest.'% (self.checkpoint_path, self.name))
            torch.save(l_data['net'].state_dict(), '%s/%s/latest' % (self.checkpoint_path, self.name))
            torch.save(l_data['net'].state_dict(),
                    '%s/%s/net_epoch_%d' % (self.checkpoint_path, self.name, l_data['epoch']))

