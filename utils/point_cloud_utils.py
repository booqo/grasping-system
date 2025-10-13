import numpy as np



def sample_cloud(
    cloud_masked, color_masked, num_point,
    remove_outliers=False,           # 是否去除离群点
    method='statistical',           # 'statistical' 或 'radius'
    nb_neighbors=30,                # 统计滤波：邻居数
    std_ratio=1.0,                  # 统计滤波：标准差阈值
    radius=0.06, min_points=8,      # 半径滤波：半径/最少邻居
    seed=1000,                       # 采样随机种子（可复现）
    keep_largest_cluster=True,   # 仅保留最大连通簇，清理“漂浮小岛”
    cluster_eps=0.015,           # DBSCAN 邻域（≈ 1~3×体素/点间距）
    cluster_min_points=60        # 簇内最少点数阈值
):
    # cloud_orig, color_orig = cloud_masked, color_masked
    
    if remove_outliers and len(cloud_masked) >= 5:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))
            if method == 'radius':
                _, inlier_idx = pcd.remove_radius_outlier(nb_points=int(min_points), radius=float(radius))
            else:
                _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
            inlier_idx = np.asarray(inlier_idx, dtype=np.int64)
            if inlier_idx.size > 0:
                cloud_masked = cloud_masked[inlier_idx]
                color_masked = color_masked[inlier_idx]
        except Exception:
            # Open3D 不可用或失败时，退化为仅采样
            pass
    
    # 2) —— 新增：DBSCAN 清理小簇，仅保留最大簇 —— 
    if keep_largest_cluster and len(cloud_masked) >= max(cluster_min_points, 5):
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))
            labels = np.asarray(
                pcd.cluster_dbscan(eps=float(cluster_eps), min_points=int(cluster_min_points), print_progress=False)
            )
            if labels.size and labels.max() >= 0:
                largest = np.bincount(labels[labels >= 0]).argmax()
                mask = labels == largest
                cloud_masked = cloud_masked[mask]
                color_masked = color_masked[mask]
        except Exception:
            pass  # 聚类不可用时跳过，不影响后续
    
    # cloud_filt, color_filt = cloud_masked, color_masked
    
    rng = np.random.default_rng(seed)

    if len(cloud_masked) >= num_point:
        idxs = rng.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = rng.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # cloud_samp, color_samp = cloud_sampled,color_sampled
    # visualize_original_filtered_sampled(cloud_orig, color_orig, cloud_filt, color_filt, cloud_samp, color_samp)
    
    return cloud_sampled, color_sampled

