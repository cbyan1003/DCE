"""
Code to find correspondances between two point clouds
"""
import torch
from pytorch3d.ops.knn import knn_points
from torch.nn.functional import cosine_similarity, log_softmax, normalize
from .model_util import nn_gather
from .hyper_emb import Gumbel_Hyper_Embedding
import gc 
import time
#@torch.jit.script
def calculate_ratio_test(dists):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Convert points so that 0 means perfect similarity and clamp to avoid numerical
    # instability
    sorted_dists, _ = torch.sort(dists, dim=1, descending=True)

    sorted_dists = (1 - sorted_dists).clamp(min=1e-9)

    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    ratio = sorted_dists[:, :, 0:1] / sorted_dists[:, :, 1:2]
    """
    I invert the weight to have higher is better, but I still don't have a very good
    intuition about what's actually being represented here. The ratio is dimensionless,
    so it's scale invariant, but it's not invariant to the distribution of features over
    the whole space. eg, if the network uses a small subspace to represent things, then
    the ratios will, on average, be lower than if it was distributed over the whole
    space, assuming that the correct correspondances feature pair only differs by a
    noise that doesn't depend on the distribution ... and who knows if that's true?

    It might make sense to consider other weighting schemes; eg, learned, or even
    ratio_test with some learned parameter. For now, I will use the simplest version of
    the ratio test; no thresholding, or reweighting.
    """
    weight = 1 - ratio

    return weight

#@torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
    idx_target = idx.gather(1, idx_source)
    return idx_source, idx_target, dist

def get_correspondences_1(
    ori_flag, P1, P2, num_corres, P1_X, P2_X, metric="cosine", ratio_test=False, gumbel_beta=0.1
):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.

    Input:
        P1          FloatTensor (N x P x F)     features for first pointcloud
        P2          FloatTensor (N x Q x F)     features for second pointcloud
        num_corres  Int                         number of correspondances
        P1_X        FloatTensor (N x P x 3)     xyz for first pointcloud
        P2_X        FloatTensor (N x Q x 3)     xyz for second pointcloud
        metric      {cosine, euclidean}         metric to be used for kNN
        ratio_test  Boolean                     whether to use ratio test for kNN

    Returns:
        LongTensor (N x 2 * num_corres)         Indices for first pointcloud
        LongTensor (N x 2 * num_corres)         Indices for second pointcloud
        FloatTensor (N x 2 * num_corres)        Weights for each correspondace
        FloatTensor (N x 2 * num_corres)        Cosine distance between features
    """
    
    if not ori_flag:
        
        euler_gamma = 0.57721566490153286060
        inv_softplus_temp = 1.2471085395024732
        softplus_scale = 1.
        
        # Calculate kNN for k=2; both outputs are (N, P, K)
        # idx_1 returns the indices of the nearest neighbor in P2
        _, idx_1, _ = knn_points(P1_X, P2_X, K=2)
        
        
        # P1 center [1,262144,256] offset [262144,2,128]; P2[idx_1.squeeze(0)] center [1,262144,3,256] offset [262144,3,2,128]
        # P2 [1,262144,256]; idx_1 [1,262144,3];
        B, _, _ = P1.center.shape
        
        # intersections_mins_1 = []
        # intersections_maxs_1 = []
        # for i in range(B):
        #     min_max = Gumbel_Hyper_Embedding.gumbel_intersection_boxs(P1.gumbel_min_offset[i], P1.gumbel_max_offset[i], P2.gumbel_min_offset[i][idx_1[i],:], P2.gumbel_max_offset[i][idx_1[i],:],gumbel_beta)
        #     intersections_mins_1.append(min_max[0]) 
        #     intersections_maxs_1.append(min_max[1])
        # del P1, P2, min_max
        # gc.collect()
        # torch.cuda.empty_cache()
        
        # intersections_mins_1 = torch.stack(intersections_mins_1, 0)
        # intersections_maxs_1 = torch.stack(intersections_maxs_1, 0)
        
        intersections_mins_1, intersections_maxs_1 = Gumbel_Hyper_Embedding.gumbel_intersection_boxs_batch(
            P1.gumbel_min_offset, #[B x N x 64]
            P1.gumbel_max_offset,
            torch.stack([P2.gumbel_min_offset[:,idx_1,:][i][i] for i in range(B)]), #[B x N x knn_num x 64]
            torch.stack([P2.gumbel_max_offset[:,idx_1,:][i][i] for i in range(B)]),
            gumbel_beta
        )

        with torch.no_grad():
            inter_vol_1 = Gumbel_Hyper_Embedding.log_soft_volume(intersections_mins_1, intersections_maxs_1, euler_gamma, inv_softplus_temp, softplus_scale, gumbel_beta)
            weights_1 = calculate_ratio_test(inter_vol_1)

        m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, num_corres)
        
        return m12_idx1, m12_idx2, m12_dist
        # max_idx_1 = torch.max(inter_vol_1, dim=1, keepdim=True)[1]
        # max_idx_2 = torch.max(inter_vol_2, dim=1, keepdim=True)[1]
        # del inter_vol_1, inter_vol_2
        # gc.collect()
        # torch.cuda.empty_cache()
        
        # idx_1 = torch.gather(idx_1.squeeze(0), 1, max_idx_1.view(-1,1))
        # idx_2 = torch.gather(idx_2.squeeze(0), 1, max_idx_2.view(-1,1))
        # del max_idx_1, max_idx_2
        # gc.collect()
        # torch.cuda.empty_cache()
    
        
        # single thread
        # fin_1 = []
        # for i, idx in enumerate(idx_1.squeeze(0)):
        #     vol = []
        #     for j in idx:
        #         intersections_min_1, intersections_max_1 = Gumbel_Hyper_Embedding.gumbel_intersection_box(P1[i], P2[j], gumbel_beta)
        #         inter_vol = Gumbel_Hyper_Embedding.log_soft_volume(intersections_min_1, intersections_max_1, euler_gamma, inv_softplus_temp, softplus_scale, gumbel_beta)
        #         vol.append(inter_vol)
        #     fin_1.append(vol.index(max(vol)))
    

    else:
        batch_size, num_points, feature_dimension = P1.shape
        assert metric in ["euclidean", "cosine"]
        if metric == "cosine":
            # Normalize points -- clamp to deal with missing points for less dense models.
            # Those points will have a cosine weight of 0.5, but will get filtered out below
            # as invalid points.
            P1 = P1 / P1.norm(dim=2, keepdim=True).clamp(min=1e-9)
            P2 = P2 / P2.norm(dim=2, keepdim=True).clamp(min=1e-9)

        if ratio_test:
            K = 2
        else:
            K = 1

        # Calculate kNN for k=2; both outputs are (N, P, K)
        # idx_1 returns the indices of the nearest neighbor in P2
        dists_1, idx_1, _ = knn_points(P1, P2, K=K)
        dists_2, idx_2, _ = knn_points(P2, P1, K=K)

        # Take the nearest neighbor for the indices for k={1, 2}
        idx_1 = idx_1[:, :, 0:1]
        idx_2 = idx_2[:, :, 0:1]

        # Transform euclidean distance of points on a sphere to cosine similarity
        cosine_1 = 1 - 0.5 * dists_1
        cosine_2 = 1 - 0.5 * dists_2

        if metric == "cosine":
            dists_1 = cosine_1
            dists_2 = cosine_2

        # Apply ratio test
        if ratio_test:
            weights_1 = calculate_ratio_test(dists_1)
            weights_2 = calculate_ratio_test(dists_2)
        else:
            weights_1 = dists_1[:, :, 0:1]
            weights_2 = dists_2[:, :, 0:1]

        # find if both the points in the correspondace are valid
        valid_z1 = P1_X[:, :, 2] != 0
        valid_z2 = P2_X[:, :, 2] != 0
        
        valid_znn1 = P2_X[:, :, 2].gather(1, idx_1.squeeze(2)) != 0.0
        valid_znn2 = P1_X[:, :, 2].gather(1, idx_2.squeeze(2)) != 0.0
        valid_1 = (valid_z1 & valid_znn1).float()
        valid_2 = (valid_z2 & valid_znn2).float()

        # multiple by valid pixels
        weights_1 = weights_1 * valid_1.unsqueeze(2)
        weights_2 = weights_2 * valid_2.unsqueeze(2)

        # Get topK matches in both directions
        m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, num_corres)
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, num_corres)

        cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
        cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

        # concatenate into correspondances and weights
        matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1).squeeze(dim=2)
        matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1).squeeze(dim=2)
        matches_dist = torch.cat((m12_dist, m21_dist), dim=1).squeeze(dim=2)
        matches_cosn = torch.cat((cosine_1, cosine_2), dim=1).squeeze(dim=2)

        return matches_idx1, matches_idx2, matches_dist, matches_cosn
    
def get_correspondences_2(
    ori_flag, P1, P2, num_corres, P1_X, P2_X, metric="cosine", ratio_test=False, gumbel_beta=0.1
):  
    
    euler_gamma = 0.57721566490153286060
    inv_softplus_temp = 1.2471085395024732
    softplus_scale = 1.
    
    start_time = time.time()
    _, idx_2, _ = knn_points(P2_X, P1_X, K=2)
    end_time = time.time()
    # print(f"##### KNN NEED: {end_time - start_time} seconds#####")
    
    B, _, C = P1.center.shape
    
    
    start_time = time.time()
    
    # intersections_mins_2 = []
    # intersections_maxs_2 = []
    # for i in range(B):
    #     min_max = Gumbel_Hyper_Embedding.gumbel_intersection_boxs(P1.gumbel_min_offset[i][idx_2[i],:], P1.gumbel_max_offset[i][idx_2[i],:], P2.gumbel_min_offset[i], P2.gumbel_max_offset[i],gumbel_beta)
    #     intersections_mins_2.append(min_max[0]) 
    #     intersections_maxs_2.append(min_max[1])
    # del P1, P2, min_max
    # gc.collect()
    # torch.cuda.empty_cache()
    # intersections_mins_2 = torch.stack(intersections_mins_2, 0)
    # intersections_maxs_2 = torch.stack(intersections_maxs_2, 0)
    
    
    
    
    intersections_mins_2, intersections_maxs_2 = Gumbel_Hyper_Embedding.gumbel_intersection_boxs_batch(
        torch.stack([P1.gumbel_min_offset[:,idx_2,:][i][i] for i in range(B)]), #[B x N x knn_num x 64]
        torch.stack([P1.gumbel_max_offset[:,idx_2,:][i][i] for i in range(B)]),
        P2.gumbel_min_offset, #[B x N x 64]
        P2.gumbel_max_offset,
        gumbel_beta
    )
    end_time = time.time()
    # print(f"##### INIT BOX NEED: {end_time - start_time} seconds#####")
    
    
    with torch.no_grad():    
        start_time = time.time()
        inter_vol_2 = Gumbel_Hyper_Embedding.log_soft_volume(intersections_mins_2, intersections_maxs_2, euler_gamma, inv_softplus_temp, softplus_scale, gumbel_beta)
        
        end_time = time.time()
        # print(f"##### BOX VOL NEED: {end_time - start_time} seconds#####")
    
        weights_2 = calculate_ratio_test(inter_vol_2)
        
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, num_corres)
    return m21_idx2, m21_idx1, m21_dist

def transfer_correspondances(corr, pc_0, pc_1, v2g_cfg, project_fn, residual):
    # get match features and coord

    corr_F_0 = nn_gather(pc_0.features_padded(), corr[0])
    corr_F_1 = nn_gather(pc_1.features_padded(), corr[1])

    # get weights for visualization
    v2g_weight = cosine_similarity(corr_F_0, corr_F_1, dim=2)
    v2g_weight = (1 + v2g_weight) / 2.0

    # project features -- corr_F_{0, 1} are B x N x F -- poiloud features
    assert project_fn is not None
    corr_F_0_p = project_fn(corr_F_0)
    corr_F_1_p = project_fn(corr_F_1)

    # calculate loss - sim_s is [0, 4]
    sim_0 = cosine_similarity(corr_F_0.detach(), corr_F_1_p, dim=2)
    sim_1 = cosine_similarity(corr_F_0_p, corr_F_1.detach(), dim=2)
    sim_s = 2 + sim_0 + sim_1

    """
    Three weighting schemes:
        - residual: incorporate residual alignment error
        - lowe: use the lowe weights from visual correpsonda
        - none: classic sim siam
    """
    if v2g_cfg.weight == "residual":
        sim_s = normalize(sim_s, p=1, dim=1)
        v2g_loss = (sim_s * residual.detach()).sum(dim=1)
    elif v2g_cfg.weight == "lowe":
        lowe_dist = normalize(corr[2].detach(), p=1, dim=1)
        v2g_loss = ((4 - sim_s) * lowe_dist).sum(dim=1)
    elif v2g_cfg.weight == "none":
        v2g_loss = 4 - sim_s.mean(dim=1)
    else:
        raise ValueError()

    return v2g_loss, v2g_weight

