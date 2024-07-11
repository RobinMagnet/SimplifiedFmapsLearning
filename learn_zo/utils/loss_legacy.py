import torch
import torch.nn as nn
class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)

class DQFMLoss(nn.Module):
    def __init__(self, w_gt=True, w_ortho=1, w_Qortho=1, w_bij=1, w_res=1, w_rank=-0.01):
        super().__init__()

        # loss HP
        self.w_gt = w_gt
        self.w_ortho = w_ortho
        self.w_Qortho = w_Qortho
        self.w_bij = w_bij
        self.w_res = w_res
        self.w_rank = w_rank
        # frob loss function
        self.frob_loss = FrobeniusLoss()

        # different losses
        self.gt_loss = 0
        self.gt_old_loss = 0
        self.ortho_loss = 0
        self.Qortho_loss = 0
        self.bij_loss = 0
        self.res_loss = 0
        self.rank_loss = 0

    def forward(self, C12_gt, C21_gt, C12, C21, C12_new, C21_new, Q12, feat1, feat2, evecs_trans1, evecs_trans2):
        loss = 0

        # gt loss (if we train on ground-truth then return directly)
        if self.w_gt:
            self.gt_old_loss = (self.frob_loss(C12, C12_gt) / self.frob_loss(0, C12_gt)  + self.frob_loss(C21, C21_gt) / self.frob_loss(0, C21_gt)) / 2
            self.gt_loss = (self.frob_loss(C12_new, C12_gt) / self.frob_loss(0, C12_gt)  + self.frob_loss(C21_new, C21_gt) / self.frob_loss(0, C21_gt)) / 2
            
            loss = self.gt_loss
            return loss

        # fmap ortho loss
        if self.w_ortho > 0:
            I = torch.eye(C12.shape[1]).unsqueeze(0).to(C12.device)
            CCt12 = C12 @ C12.transpose(1, 2)
            CCt21 = C21 @ C21.transpose(1, 2)
            
            CCt12_new = C12_new @ C12_new.transpose(1, 2)
            CCt21_new = C21_new @ C21_new.transpose(1, 2)
            self.ortho_loss = (self.frob_loss(CCt12, I) + self.frob_loss(CCt21, I) + self.frob_loss(CCt12_new, I) + self.frob_loss(CCt21_new, I)) * self.w_ortho / 2
            # self.ortho_loss = (self.frob_loss(CCt12, I) + self.frob_loss(CCt21, I)) * self.w_ortho
            loss += self.ortho_loss

        # fmap bij loss
        if self.w_bij > 0:
            I = torch.eye(C12.shape[1]).unsqueeze(0).to(C12.device)
            self.bij_loss = (self.frob_loss(torch.bmm(C12, C21), I) + self.frob_loss(torch.bmm(C21, C12), I)) * self.w_bij
            # loss += self.bij_loss

        if self.w_res > 0:
            self.res_loss = self.frob_loss(C12, C12_new) + self.frob_loss(C21, C21_new)
            self.res_loss *= self.w_res
            loss += self.res_loss

        if self.w_rank < 0:
            F_hat = torch.bmm(evecs_trans1, feat1)
            G_hat = torch.bmm(evecs_trans2, feat2)
            F = F_hat @ F_hat.transpose(1, 2)
            G = G_hat @ G_hat.transpose(1, 2)
            I = torch.eye(F.shape[1]).unsqueeze(0).to(F.device)
            rank_pen = 0
            for i in range(F_hat.shape[0]):
                rank_pen += F_hat[i].norm(p='nuc') + G_hat[i].norm(p='nuc')
            self.rank_loss = rank_pen
            #self.rank_loss = self.frob_loss(F+G, 2*I)
            self.rank_loss *= self.w_rank
            # loss += self.rank_loss

        # qfmap ortho loss
        if Q12 is not None and self.w_Qortho > 0:
            I = torch.eye(Q12.shape[1]).unsqueeze(0).to(Q12.device)
            CCt = Q12 @ torch.conj(Q12.transpose(1, 2))
            self.Qortho_loss = self.frob_loss(CCt, I) * self.w_Qortho
            loss += self.Qortho_loss

        return [loss, self.gt_old_loss, self.gt_loss, self.ortho_loss, self.bij_loss, self.res_loss, self.rank_loss]