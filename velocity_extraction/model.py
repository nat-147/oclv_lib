import torch
from torch import nn

from velocity_extraction.utils import mse_loss_fn, cosine_dissim


class VelocityExtractor(nn.Module):
    def __init__(self, slot_dim, v_dim, loss_weights, num_slots, use_input_pairs=False):
        super().__init__()

        self.loss_weights = loss_weights
        self.use_input_pairs = use_input_pairs
        
        hidden_dim = 4*slot_dim

        self.get_v = nn.Sequential(
                nn.Linear(2*slot_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, v_dim),
                nn.Tanh(),
                )
        
        self.combine = nn.Sequential(
            nn.Linear(slot_dim + v_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )


    def encode(self, z1, z2):
        v = self.get_v(torch.cat([z1, z2], dim=-1))

        return v

    def decode(self, v, z1): 
        z2 = self.combine(torch.cat([z1, v], dim=-1)) + z1

        return z2
        

    def isometry_loss_fn(self, pred_vs, similarity_img1, similarity_img2, threshold=1e-3, weight=1e-2):
        if pred_vs.numel() == 0:
            return 0

        isometry_numerator = torch.norm(pred_vs, p=2, dim=-1).flatten()

        isometry_denominator = cosine_dissim(
            similarity_img1,
            similarity_img2,
        ).flatten()
        
        close_points = isometry_denominator < threshold

        if close_points.sum() > 1:
            isometry_loss = weight * torch.var(isometry_numerator[close_points])
        else:
            isometry_loss = torch.tensor(0).float().cuda()

        if isometry_loss.isnan().item():
            breakpoint()

        return isometry_loss.squeeze()

    
    def compute_loss(self, true_z, loss_weights):       
        true_z1, true_z2, true_z3 = true_z[:,0:-1:2,:,:], true_z[:,1:-1:2,:,:], true_z[:,2::2,:,:]
        
        zm = true_z.mean(dim=-2, keepdim=True)
        zm1, zm2 = zm[:,0:-1:2,:,:], zm[:,1:-1:2,:,:]
                    
        pred_v_z1_z2 = self.encode(zm1, zm2).expand(-1,-1, true_z.shape[-2], -1) 

        #pred_v_z1_z2 = self.encode(true_z1, true_z2)
        
        pred_z3 = self.decode(pred_v_z1_z2, true_z2)
                

        # NOTE: pred_vs[:, :-1, :] describes inferred velocities from i2 -> i3 and pred_vs[:, 1:, :] describes inferred velocities from i3 -> i4
        '''
        image prediction loss
        '''
        z_pred_loss = mse_loss_fn(pred_z3, true_z3, weight=loss_weights['z_pred_loss_weight'])


        '''
        loop closure loss
        '''      
        pred_v_z1_z2_sum = pred_v_z1_z2.sum(dim=1)
        loop_loss =  mse_loss_fn(pred_v_z1_z2_sum, torch.zeros_like(pred_v_z1_z2_sum).cuda(), weight=loss_weights['loop_loss_weight'])


        '''
        shortcut estimation loss
        '''
        true_z4 = true_z2[:, 1:]
        pred_v_z2_z4 = pred_v_z1_z2[:, :-1, :, :] + pred_v_z1_z2[:, 1:, :, :]
        
        shortcut_loss = (
            
            mse_loss_fn(
                self.decode(pred_v_z2_z4, true_z2[:, :-1, :, :]),
                true_z4,
                weight=loss_weights['shortcut_loss_weight'],
                ) + \
            mse_loss_fn(
                self.decode(-pred_v_z2_z4, true_z4),
                true_z2[:, :-1, :, :],
                weight=loss_weights['shortcut_loss_weight'],
                )
        )

        '''
        isometry loss
        '''
        isometry_loss = self.isometry_loss_fn(
            pred_v_z1_z2, true_z1, true_z2,
            threshold=loss_weights['isometry_similarity_threshold'],
            weight=loss_weights['isometry_loss_weight'],
            )

        loss_dict = {
            'z_pred_loss' : z_pred_loss,
            'loop_loss' : loop_loss,
            'shortcut_loss' : shortcut_loss,
            'isometry_loss' : isometry_loss,
        }
        
        return loss_dict, pred_z3


    def compute_loss_pairs(self, true_z, loss_weights):
        true_z1, true_z2 = true_z[:, :-1, :, :], true_z[:, 1:, :, :]

        pred_vs_z1_z2 = self.encode(true_z)
        pred_z2 = self.decode(pred_vs_z1_z2, true_z1)

        true_z3 = true_z2[:, 1:]
        pred_vs_z1_z3 = pred_vs_z1_z2[:, :-1, :, :] + pred_vs_z1_z2[:, 1:, :, :]


        # NOTE: pred_vs[:, :-1, :] describes inferred velocities from i2 -> i3 and pred_vs[:, 1:, :] describes inferred velocities from i3 -> i4
        '''
        image prediction loss
        '''
        z_pred_loss = mse_loss_fn(pred_z2, true_z2, weight=loss_weights['z_pred_loss_weight'])
        
        '''
        loop closure loss
        '''
        pred_vs_z1_z2_sum = pred_vs_z1_z2.sum(dim=1)

        loop_loss = mse_loss_fn(pred_vs_z1_z2_sum, torch.zeros_like(pred_vs_z1_z2_sum).cuda(), weight=loss_weights['loop_loss_weight'])

        '''
        shortcut estimation loss
        '''      
        shortcut_loss = (
            
            mse_loss_fn(
                self.decode(pred_vs_z1_z3, true_z1[:, :-1, :, :]),
                true_z3,
                weight=loss_weights['shortcut_loss_weight'],
                ) + \
            mse_loss_fn(
                self.decode(-pred_vs_z1_z3, true_z3),
                true_z1[:, :-1, :, :],
                weight=loss_weights['shortcut_loss_weight'],
                )
        )
        
        '''
        isometry loss
        '''
        isometry_loss = self.isotropy_loss_fn(
            pred_vs_z1_z2, true_z1, true_z2,
            
            threshold=loss_weights['isometry_similarity_threshold'],
            weight=loss_weights['isometry_loss_weight'],
            )

        loss_dict = {
            'z_pred_loss' : z_pred_loss,
            'loop_loss' : loop_loss,
            'shortcut_loss' : shortcut_loss,
            'isometry_loss' : isometry_loss,
        }

        return loss_dict, pred_z2



    def forward(self, z):
        if self.use_input_pairs:
            losses, pred_z = self.compute_loss_pairs(z, self.loss_weights)
        else:
            losses, pred_z = self.compute_loss(z, self.loss_weights)

        return losses, pred_z