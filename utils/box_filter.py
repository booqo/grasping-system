import numpy as np
import open3d as o3d
def _axis_from_string(open_axis: str):
    """
    将 open_axis (如 '+z','-y') 解析为: (轴索引k, 符号sgn, 单位向量v_open)
      轴索引: x->0, y->1, z->2
      sgn: +1 / -1, 表示开口朝向
      v_open: 长度为1的numpy向量
    """
    open_axis = open_axis.strip().lower()
    assert open_axis in {"+x","-x","+y","-y","+z","-z"}
    sgn = +1 if open_axis[0] == '+' else -1
    ax  = open_axis[1]
    k = {'x':0,'y':1,'z':2}[ax]
    v = np.zeros(3, dtype=np.float64); v[k] = float(sgn)
    return k, sgn, v

# ---------- 辅助：矩阵求逆 ----------
def _invert_transform(T: np.ndarray):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -R.T @ t
    return Ti

# ---------- 辅助：可选仅 yaw 投影（将旋转约束为绕Z轴） ----------
def _project_yaw_only(R):
    # 若您的“竖直”并非模型Z，请按需改写
    yaw = np.arctan2(R[1,0], R[0,0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy,-sy,0],
                   [sy, cy,0],
                   [ 0,  0,1]], dtype=np.float64)
    return Ry

# ---------- 辅助：口沿折线采样（统一实现，覆盖 ±x/±y/±z） ----------
def _sample_open_rim(size_xyz, open_axis, spacing):
    """
    在开口所在平面（坐标轴 k = ±half）上，沿其余两个轴围出矩形“口沿”折线。
    """
    sx, sy, sz = map(float, size_xyz)
    halves = np.array([sx/2, sy/2, sz/2], dtype=np.float64)
    k, sgn, _ = _axis_from_string(open_axis)

    # 其余两个轴索引
    axes = [0,1,2]; axes.remove(k)
    a, b = axes  # 在 (a,b) 平面内形成矩形

    # 每个方向的取样数（至少4个点）
    na = max(4, int(np.ceil(sx/spacing)) if a==0 else int(np.ceil(sy/spacing)) if a==1 else int(np.ceil(sz/spacing)))
    nb = max(4, int(np.ceil(sx/spacing)) if b==0 else int(np.ceil(sy/spacing)) if b==1 else int(np.ceil(sz/spacing)))

    ra = np.linspace(-halves[a], +halves[a], na)
    rb = np.linspace(-halves[b], +halves[b], nb)

    # 固定开口平面坐标
    fixed = sgn * halves[k]

    # 四条边（按顺时针拼接）
    loop = []
    # 边1：a 变、b = -hb
    P1 = np.zeros((na,3), dtype=np.float64); P1[:,a] = ra; P1[:,b] = -halves[b]; P1[:,k] = fixed
    # 边2：b 变、a = +ha
    P2 = np.zeros((nb,3), dtype=np.float64); P2[:,a] = +halves[a]; P2[:,b] = rb;  P2[:,k] = fixed
    # 边3：a 变（逆序）、b = +hb
    P3 = np.zeros((na,3), dtype=np.float64); P3[:,a] = ra[::-1]; P3[:,b] = +halves[b]; P3[:,k] = fixed
    # 边4：b 变（逆序）、a = -ha
    P4 = np.zeros((nb,3), dtype=np.float64); P4[:,a] = -halves[a]; P4[:,b] = rb[::-1]; P4[:,k] = fixed
    loop = np.concatenate([P1,P2,P3,P4], axis=0)
    return loop

# ---------- 辅助：几何门控（仅保留靠近四面内壁，且远离底面） ----------
def _gate_points_near_walls(
    pts_cam, T_model_to_cam, size_xyz, open_axis, band=0.015
):
    """
    将目标点云变换到“模型坐标系”，仅保留：
      1) 接近四面内壁的带状区域；2) 远离“底面”（开口反向的那一面）。
    band：带宽（米），建议随尺度或体素大小调整。
    """
    Ti = _invert_transform(T_model_to_cam)
    Rm2c = Ti[:3,:3]; tm2c = Ti[:3,3]
    pts_m = (Rm2c @ pts_cam.T).T + tm2c

    sx, sy, sz = map(float, size_xyz)
    hx, hy, hz = sx/2, sy/2, sz/2
    halves = np.array([hx, hy, hz], dtype=np.float64)

    # 基本包络（略放宽）
    inside = (np.abs(pts_m[:,0]) <= hx + band) & \
             (np.abs(pts_m[:,1]) <= hy + band) & \
             (np.abs(pts_m[:,2]) <= hz + band)

    k, sgn, _ = _axis_from_string(open_axis)
    coords = [pts_m[:,0], pts_m[:,1], pts_m[:,2]]

    # 接近四面内壁（在非开口轴的两个方向，各自接近 ±half）
    near_walls = np.zeros(pts_m.shape[0], dtype=bool)
    for axis in (i for i in range(3) if i != k):
        near_pos = np.abs(np.abs(coords[axis]) - halves[axis]) <= band
        near_walls |= near_pos

    # 排除“底面”附近带：底面位于 k 轴的 -sgn 方向半长处
    # 具体阈值：距底面 > band
    not_bottom = np.ones(pts_m.shape[0], dtype=bool)
    if k == 0:   # x轴开口
        if sgn > 0:  # 底面在 x = -hx
            not_bottom = coords[0] > (-hx + band)
        else:        # 底面在 x = +hx
            not_bottom = coords[0] < ( hx - band)
    elif k == 1: # y轴开口
        if sgn > 0:
            not_bottom = coords[1] > (-hy + band)
        else:
            not_bottom = coords[1] < ( hy - band)
    else:        # z轴开口
        if sgn > 0:
            not_bottom = coords[2] > (-hz + band)
        else:
            not_bottom = coords[2] < ( hz - band)

    mask = inside & near_walls & not_bottom
    return mask

# ---------- 工具：反投影到相机坐标 ----------
def backproject_to_cam(depth_m, K):
    """ depth_m: HxW (meters), K: 3x3 """
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m.reshape(-1)
    x = (u.reshape(-1) - K[0,2]) * z / K[0,0]
    y = (v.reshape(-1) - K[1,2]) * z / K[1,1]
    pts = np.stack([x, y, z], axis=1)  # (N,3), 相机坐标系
    return pts

def to_homo(pts):
    return np.c_[pts, np.ones((pts.shape[0], 1))]

def generate_open_box_interior_points(size_xyz, spacing=0.01, open_axis="+z"):
    """
    生成开口箱体“内部 5 面”的离散点与法向（模型坐标系，箱体中心在原点）。
    size_xyz: (Lx, Ly, Lz) 外尺寸（米）
    spacing : 每个面的采样间距（米）
    open_axis ∈ {+x,-x,+y,-y,+z,-z} 指定开口朝向
    返回:
      pts_m: (N,3)  模型系点
      nrm_m: (N,3)  对应法向（指向“箱体内部”）
    """
    Lx, Ly, Lz = map(float, size_xyz)
    hx, hy, hz = Lx/2, Ly/2, Lz/2

    # 定义 6 个面的中心、平面两轴(ux, uy)、法向(指向内部)，以及开口面
    faces = {
        "+x": (np.array([+hx,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([-1,0,0]), Ly, Lz),
        "-x": (np.array([-hx,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([+1,0,0]), Ly, Lz),
        "+y": (np.array([0,+hy,0]), np.array([1,0,0]), np.array([0,0,1]), np.array([0,-1,0]), Lx, Lz),
        "-y": (np.array([0,-hy,0]), np.array([1,0,0]), np.array([0,0,1]), np.array([0,+1,0]), Lx, Lz),
        "+z": (np.array([0,0,+hz]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,-1]), Lx, Ly),
        "-z": (np.array([0,0,-hz]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,+1]), Lx, Ly),
    }
    # 去掉“开口面”
    faces.pop(open_axis, None)

    pts_all, nrm_all = [], []
    for _, (c, ux, uy, n, sx, sy) in faces.items():
        nx = max(int(np.ceil(sx/spacing)), 2)
        ny = max(int(np.ceil(sy/spacing)), 2)
        xs = np.linspace(-sx/2, +sx/2, nx)
        ys = np.linspace(-sy/2, +sy/2, ny)
        grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis=-1).reshape(-1,2)
        P = c[None,:] + grid[:,0:1]*ux[None,:] + grid[:,1:2]*uy[None,:]
        N = np.tile(n, (P.shape[0], 1))
        pts_all.append(P); nrm_all.append(N)

    pts_m = np.concatenate(pts_all, axis=0)
    nrm_m = np.concatenate(nrm_all, axis=0)
    return pts_m.astype(np.float64), nrm_m.astype(np.float64)

def init_pose_from_cam(pts_cam, use_pca_yaw=True):
    """
    初始化 R,t：令箱体局部 +Z 与相机 +Z 对齐；t 为点云中位数；
    若 use_pca_yaw=True，则在 x-y 平面用 PCA 估计 yaw。
    返回 R(3x3), t(3,)
    """
    z_cam = -(np.array([0.,0.,1.], dtype=np.float64) )     # 相机 +Z 指向前方/箱内
    center = np.median(pts_cam, axis=0)

    if use_pca_yaw:
        P = pts_cam - center
        Q = P.copy(); Q[:,2] = 0.0                      # 投影到 x-y 平面
        if np.allclose(Q.var(axis=0).sum(), 0, atol=1e-9):
            x_axis = np.array([1.,0.,0.])
        else:
            U,S,Vt = np.linalg.svd(Q, full_matrices=False)
            x_axis = Vt[0]
            x_axis[2] = 0.0
            x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    else:
        x_axis = np.array([1.,0.,0.])

    z_axis = z_cam
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = np.array([0.,1.,0.])
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R, center

def align_open_box_to_pts_cam(
    pts_cam,                      # (N,3) 相机系点云
    size_xyz=(0.40,0.30,0.22),    # 初始外尺寸 (Lx,Ly,Lz)
    open_axis="+z",
    spacing=0.01,                 # 面上采样间距
    icp_dist=0.02,                # ICP 最大对应距离
    outer_iters=4,                # 外层迭代次数
    use_pca_yaw=True,
    estimate_target_normals=True, # 为观测点云估计法向（推荐）
):
    pts_cam = np.asarray(pts_cam, dtype=np.float64)

    # 1) 生成模型点与法向（模型系）
    pts_m, nrm_m = generate_open_box_interior_points(size_xyz, spacing, open_axis)

    # 2) 初始化 R,t（模型->相机）
    R, t = init_pose_from_cam(pts_cam, use_pca_yaw=use_pca_yaw)

    # 3) 构建 Open3D 点云
    src_model = o3d.geometry.PointCloud()
    src_model.points  = o3d.utility.Vector3dVector(pts_m)
    src_model.normals = o3d.utility.Vector3dVector(nrm_m)

    tgt_obs = o3d.geometry.PointCloud()
    tgt_obs.points = o3d.utility.Vector3dVector(pts_cam)
    if estimate_target_normals:
        tgt_obs.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(spacing*3, icp_dist*0.8), max_nn=50)
        )

    # 4) 外层迭代（每次把模型放到当前位姿，在相机系下与观测做 point-to-plane ICP）
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    est_pp = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    for _ in range(outer_iters):
        # 模型 -> 相机系
        src_in_cam = o3d.geometry.PointCloud(src_model)
        src_in_cam.transform(T)

        # 目标：观测（法向更噪，但可用）或把“模型”当目标（下行两种二选一）
        reg = o3d.pipelines.registration.registration_icp(
            src_in_cam, tgt_obs, icp_dist, np.eye(4), est_pp, criteria
        )
        # reg.transformation 使“源”更贴近“目标”，即：T_delta ∘ src_in_cam ≈ tgt_obs
        T_delta = reg.transformation
        # 更新全局位姿：先应用旧 T，再左乘增量
        T = T_delta @ T

    # 5) 输出 OBB（相机系）与位姿
    obb = o3d.geometry.OrientedBoundingBox(center=T[:3,3], R=T[:3,:3], extent=np.asarray(size_xyz, dtype=np.float64))
    return T, obb

def align_open_box_pose_only(
    pts_cam,                      # (N,3) 相机系点云（建议已做统计滤波）
    size_xyz=(0.40, 0.30, 0.22),  # 已知/可信的箱体外尺寸
    open_axis="+z",
    spacing=0.01,                 # 模型面采样间距
    voxel_stages=(0.02, 0.01, 0.005),   # 目标点云多尺度体素
    dist_stages=(0.05, 0.03, 0.015),    # 对应 ICP 距离阈值
    use_pca_yaw=True
):
    pts_cam = np.asarray(pts_cam, dtype=np.float64)

    # 模型点与法向（源）
    pts_m, nrm_m = generate_open_box_interior_points(size_xyz, spacing, open_axis)
    src = o3d.geometry.PointCloud()
    src.points  = o3d.utility.Vector3dVector(pts_m)
    src.normals = o3d.utility.Vector3dVector(nrm_m)  # point-to-plane 需要源或目标法向之一

    # 目标点云
    tgt_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))

    # 初始化
    R, t = init_pose_from_cam(pts_cam, use_pca_yaw=use_pca_yaw)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t

    est_pp = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # 多尺度仅对目标点云下采样与法向估计；源保持稠密且保留法向
    for vox, dmax in zip(voxel_stages, dist_stages):
        tgt = tgt_full.voxel_down_sample(voxel_size=vox)
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vox*3.0, max_nn=60)
        )
        reg = o3d.pipelines.registration.registration_icp(
            source=src, target=tgt,
            max_correspondence_distance=float(dmax),
            init=T,  # 直接把当前估计作为初值
            estimation_method=est_pp,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
        )
        T = reg.transformation  # 更新位姿

    # 固定尺寸构造 OBB（相机系）
    obb = o3d.geometry.OrientedBoundingBox(center=T[:3,3], R=T[:3,:3], extent=np.asarray(size_xyz, dtype=np.float64))
    return T, obb

def align_open_box_pose_without_bottom(
    pts_cam,                      # (N,3) 相机系点云（建议已做统计滤波）
    size_xyz=(0.40, 0.30, 0.22),  # 已知/可信的箱体外尺寸
    open_axis="+z",
    spacing=0.01,                 # 模型面/口沿采样间距
    voxel_stages=(0.02, 0.01, 0.005),
    dist_stages=(0.05, 0.03, 0.015),
    use_pca_yaw=True,
    lock_yaw_only=False,          # 可选：将旋转投影为仅 yaw（若有地平约束）
    wall_band_ratio=0.04,         # 内壁带宽占最小尺寸比例（几何门控）
    rim_weight=5                  # 末端口沿点的重复倍数（相当于加权）
):
    """
    仅依赖箱体内壁（**不含底面**）与口沿进行位姿求解；显式避免底面参与匹配：
      1) 多尺度 ICP：point-to-plane（源=内壁，目标=门控后实测点）
      2) 末端精修：口沿折线 point-to-point，锁定沿开口轴方向的平移
    复用：
      - init_pose_from_cam(pts_cam, use_pca_yaw)
      - generate_open_box_interior_points(size_xyz, spacing, open_axis)
    返回：
      T (4x4), obb (Open3D OrientedBoundingBox，extent=size_xyz)
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    sx, sy, sz = map(float, size_xyz)
    wall_band = wall_band_ratio * min(sx, sy, sz)

    # ---------- 源：内壁点云（复用 generate_open_box_interior_points），并显式去掉“底面” ----------
    pts_m_all, nrm_m_all = generate_open_box_interior_points(size_xyz, spacing, open_axis)
    # 利用法向与开口方向对齐关系来剔除“底面”：nrm · v_open ≈ +1
    _, _, v_open = _axis_from_string(open_axis)
    cos_aln = (nrm_m_all @ v_open.reshape(3,))  # 与开口方向的夹角余弦
    keep_mask = cos_aln < 0.99  # 过滤掉与开口方向同向的那一面（即底面）
    pts_m = pts_m_all[keep_mask]
    nrm_m = nrm_m_all[keep_mask]

    src_walls = o3d.geometry.PointCloud()
    src_walls.points  = o3d.utility.Vector3dVector(pts_m)
    src_walls.normals = o3d.utility.Vector3dVector(nrm_m)  # point-to-plane 需要源或目标法向之一

    # ---------- 目标：完整点云（逐层门控/体素） ----------
    tgt_full = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))

    # ---------- 初始化（复用 init_pose_from_cam） ----------
    R, t = init_pose_from_cam(pts_cam, use_pca_yaw=use_pca_yaw)
    if lock_yaw_only:
        R = _project_yaw_only(R)
    T = np.eye(4, dtype=np.float64); T[:3,:3] = R; T[:3,3] = t

    # ---------- 多尺度 ICP（point-to-plane），每层先几何门控再估法向 ----------
    est_ptpl = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    assert len(voxel_stages) == len(dist_stages)
    for vox, dmax in zip(voxel_stages, dist_stages):
        # 几何门控：仅保留靠近四面内壁的点，且远离底面
        band = max(wall_band, vox)  # 带宽不小于体素
        mask = _gate_points_near_walls(
            pts_cam, T_model_to_cam=T, size_xyz=size_xyz, open_axis=open_axis, band=band
        )
        pts_gate = pts_cam[mask]
        # 若门控过严导致点数过少，回退使用全局点云
        if pts_gate.shape[0] < 300:
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
        else:
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_gate))

        tgt = tgt.voxel_down_sample(voxel_size=vox)
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vox*3.0, max_nn=60)
        )

        reg = o3d.pipelines.registration.registration_icp(
            source=src_walls, target=tgt,
            max_correspondence_distance=float(dmax),
            init=T,
            estimation_method=est_ptpl,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
        )
        T = np.array(reg.transformation, dtype=np.float64, copy=True)
        if lock_yaw_only:
            T[:3,:3] = _project_yaw_only(T[:3,:3])

    # ---------- 末端精修：口沿折线 point-to-point（提高沿开口轴的可观测性） ----------
    rim = _sample_open_rim(size_xyz, open_axis, spacing=max(spacing, voxel_stages[-1]))
    if rim_weight > 1:
        rim = np.repeat(rim, repeats=int(rim_weight), axis=0)
    src_rim = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rim))

    est_ptpt = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    reg_rim = o3d.pipelines.registration.registration_icp(
        source=src_rim, target=tgt,  # 复用上一层的目标点云
        max_correspondence_distance=float(dist_stages[-1] * 1.2),
        init=T,
        estimation_method=est_ptpt,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    )
    T = np.array(reg_rim.transformation, dtype=np.float64, copy=True)
    if lock_yaw_only:
        T[:3,:3] = _project_yaw_only(T[:3,:3])

    # ---------- 构造固定尺寸 OBB（相机系） ----------
    obb = o3d.geometry.OrientedBoundingBox(
        center=T[:3,3], R=T[:3,:3], extent=np.asarray(size_xyz, dtype=np.float64)
    )
    return T, obb

def visualize_alignment_cam(pts_cam, T_m2c, size_xyz, open_axis="+z", spacing=0.01):
    # 观测点
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam.astype(np.float64)))
    pcd.paint_uniform_color([0.6,0.6,0.7])

    # 模型内部 5 面
    pts_m, _ = generate_open_box_interior_points(size_xyz, spacing, open_axis)
    model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_m))
    model.paint_uniform_color([1.0,0.2,0.2])
    model.transform(T_m2c)

    # 线框 OBB
    obb = o3d.geometry.OrientedBoundingBox(center=T_m2c[:3,3], R=T_m2c[:3,:3], extent=np.asarray(size_xyz))
    # corners = np.asarray(obb.get_box_points())
    # edges = np.array([[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]])
    # ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners),
    #                           lines=o3d.utility.Vector2iVector(edges))
    # ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0,1,0.]]),(edges.shape[0],1)))
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([
        pcd, model, ls,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    ])


def _obb_to_mesh(obb: o3d.geometry.OrientedBoundingBox, color=(1.0, 0.6, 0.0)):
    """
    将 OBB 转为实体盒网格用于展示：按 obb.extent 创建 box，并用 (R, center) 施加位姿。
    """
    ex, ey, ez = [float(v) for v in obb.extent]
    mesh = o3d.geometry.TriangleMesh.create_box(width=ex, height=ey, depth=ez)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    # 平移到局部原点（OBB 默认中心在原点）
    mesh.translate(-np.array([ex, ey, ez]) / 2.0)
    # 应用 R,t
    T = np.eye(4)
    T[:3, :3] = np.asarray(obb.R)
    T[:3,  3] = np.asarray(obb.center)
    mesh.transform(T)
    return mesh

def _make_path_lines(points: np.ndarray, color=(0.2, 0.8, 1.0)):
    """
    将若干路径点连成折线进行可视化。
    points: (K,3)
    """
    if points is None or len(points) < 2:
        return None
    pts = o3d.utility.Vector3dVector(points.astype(np.float64))
    lines = [[i, i+1] for i in range(points.shape[0]-1)]
    ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color(color)
    return ls

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def mat_to_quat(R):
    """3x3 -> [w,x,y,z]"""
    R = np.asarray(R, dtype=float)
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))

def quat_to_mat(q):
    """[w,x,y,z] -> 3x3"""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ], dtype=float)
    return R

def quat_mul(q1, q2):
    """[w,x,y,z] x [w,x,y,z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return quat_normalize(np.array([w, x, y, z], dtype=float))

def quat_from_two_vectors(a, b):
    """最小转角：把单位向量 a 旋到 b（a,b in R^3）"""
    a = np.asarray(a, dtype=float); a /= (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, dtype=float); b /= (np.linalg.norm(b) + 1e-12)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # 零旋转
    if dot < -1.0 + 1e-8:
        # 反向：找一个与 a 不共线的轴
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:                 # a 太贴近 x，就用另一个基
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = axis - a * np.dot(a, axis)   # 去掉与 a 平行分量
        axis /= (np.linalg.norm(axis) + 1e-12)
        # 180° 旋转，w=0
        return np.array([0.0, *axis], dtype=float)
    # 一般情形
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    w = 0.5 * s
    v = axis / s
    return quat_normalize(np.array([w, v[0], v[1], v[2]], dtype=float))

def _rot_between_vecs_quat(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-8:          # 平行：单位旋转
        return np.eye(3)
    if c < -1.0 + 1e-8:         # 反向：绕任意正交轴180°
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(a, axis)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        w = 0.0
    else:
        v = np.cross(a, b)
        w = np.sqrt((1.0 + c) * 0.5)
        x, y, z = (v / (2.0 * w))
    # quat -> R
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [1 - 2*(yy+zz),   2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx+yy)]
    ], dtype=float)
    return R

# ---------- 主类：箱体 OBB 过滤 ----------
class BoxOBBFilter:
    """
    用箱体的 Oriented Bounding Box（带安全膨胀）过滤抓取：
    1) 指爪/掌与箱体 OBB 相交则剔除
    2) 预抓取→抓取的接近路径穿过 OBB 则剔除
    3) 距箱口边缘过近（rim_band）则剔除
    """
    def __init__(self, K, T_cam_base=np.eye(4),
                 finger_t=0.01, finger_len=0.06,
                 expand_delta=0, rim_band=0.02,wall_thickness=0.005,
                 debug=False,
                 max_print_grasps=1000,
                 aling_grasp=True ):
        
        self.K = K.astype(np.float64)
        self.T_cam_base = T_cam_base.astype(np.float64)
        self.finger_t = float(finger_t)     # 指爪厚度
        self.finger_len = float(finger_len) # 指爪长度（沿接近方向）
        self.expand_delta = float(expand_delta) # OBB 膨胀量（安全裕度）
        self.rim_band = float(rim_band)     # 箱沿禁带宽度
        self.obb = None
        self.obb_forbid = None
        self.top_plane = None   # (n, p0)
        self.wall_thickness = float(wall_thickness)
        self.forbid_slabs = None   # 5 个薄板 OBB 列表（侧四+底）
        self.debug = bool(debug)
        self.max_print_grasps = int(max_print_grasps)
        self.aling_grasp = bool(aling_grasp)

     # --------- 调试工具函数（格式化/正交性检查） ----------
    
    @staticmethod
    def _fmt(v, prec=4):
        return np.array2string(np.asarray(v), formatter={'float_kind':lambda x: f'{x:.{prec}f}'})

    @staticmethod
    def _rot_ortho_stats(R):
        RtR = R.T @ R
        I = np.eye(3)
        err = np.max(np.abs(RtR - I))
        det = np.linalg.det(R)
        return err, det

    def _print_frame_stats(self, name, R, t):
        err, det = self._rot_ortho_stats(R)
        print(f"[DBG] {name}: t = {self._fmt(t)}, det(R)={det:.6f}, ortho_err={err:.2e}")

    def _print_aabb(self, tag, geom):
        aabb = geom.get_axis_aligned_bounding_box()
        print(f"[DBG] AABB {tag}: min={self._fmt(aabb.get_min_bound())}, max={self._fmt(aabb.get_max_bound())}")

    def build(self, pts_cam):

        # 1) 打印 T_cam_base 关键信息
        if self.debug:
            print("\n========== [BUILD DEBUG] ==========")
            self._print_frame_stats("T_cam_base (R|t)", self.T_cam_base[:3,:3], self.T_cam_base[:3,3])

        if pts_cam.shape[0] < 100:
            raise ValueError("BoxOBBFilter: 有效点过少，检查 mask 或深度。")

        # 相机→基座,得到基体坐标系下的点云
        pts_base = (to_homo(pts_cam) @ self.T_cam_base.T)[:, :3]

        if self.debug:
            print(f"[DBG] pts_cam centroid = {self._fmt(np.median(pts_cam, axis=0))}")
            print(f"[DBG] pts_base centroid = {self._fmt(np.median(pts_base, axis=0))}")


        # 点云 → OBB（+ 去噪可选）
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_base))
        pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)[0]

        # T_m2c, obb = align_open_box_to_pts_cam(
        #                             pts_cam,
        #                             size_xyz=(0.365, 0.265, 0.168),   # 你的箱体外尺寸初值
        #                             open_axis="+z",
        #                             spacing=0.01,
        #                             icp_dist=0.02,
        #                             outer_iters=4,
        #                             use_pca_yaw=True
        #                             )
        T_m2c, obb_cam = align_open_box_pose_only(            
                                    pts_cam,                      # (N,3) 相机系点云
                                    size_xyz=(0.365, 0.265, 0.168), # 已知/可信的箱体外尺寸
                                    open_axis="+z",
                                    spacing=0.01,                 # 模型面采样间距
                                    voxel_stages=(0.02, 0.01, 0.005),   # 目标点云多尺度体素下
                                    dist_stages=(0.05, 0.03, 0.015),    # 对应 ICP 距离阈值
                                    use_pca_yaw=True)
        # T_m2c, obb_cam = align_open_box_pose_without_bottom(            
        #                             pts_cam,                      # (N,3) 相机系点云
        #                             size_xyz=(0.365, 0.265, 0.122), # 已知/可信的箱体外尺寸
        #                             open_axis="+z",
        #                             spacing=0.01,                 # 模型面采样间距
        #                             voxel_stages=(0.02, 0.01, 0.005),   # 目标点云多尺度体素下
        #                             dist_stages=(0.05, 0.03, 0.015),    # 对应 ICP 距离阈值
        #                             use_pca_yaw=True,
        #                             lock_yaw_only=True)
        

        if self.debug:
            print(f"[DBG] T_model_to_cam:\n{self._fmt(T_m2c)}")
            self._print_frame_stats("R_model_to_cam", T_m2c[:3,:3], T_m2c[:3,3])
            print(f"[DBG] obb_cam.center={self._fmt(obb_cam.center)}, extent={self._fmt(obb_cam.extent)}")
        # visualize_alignment_cam(pts_cam, T_m2c, size_xyz=(0.365, 0.265, 0.168), open_axis="+z", spacing=0.01)

        # obb = pcd.get_oriented_bounding_box()
        obb_base = self._transform_obb_to_base(obb_cam)
        self.obb = obb_base
        self.obb_forbid = self._expand_obb(self.obb, self.expand_delta)
        self.forbid_slabs = self._make_wall_slabs(self.obb, wall_t=self.wall_thickness,
                                          expand=self.expand_delta)
        
        self.bottom_normal_base = -self.obb.R[:, 2] / (np.linalg.norm(self.obb.R[:, 2]) + 1e-12)
            
        if self.debug:
            print(f"[DBG] obb_base.center={self._fmt(self.obb.center)}, extent={self._fmt(self.obb.extent)}")
            self._print_aabb("obb_forbid(base)", self.obb_forbid)
            # 打印第一块薄板
            if len(self.forbid_slabs) > 0:
                self._print_aabb("forbid_slabs[0](base)", self.forbid_slabs[0])

        # 估计箱口法向
        R = self.obb.R
        ext = self.obb.extent
        n_top = R[:, 2] / np.linalg.norm(R[:, 2])
        p0 = self.obb.center + n_top * (ext[2] / 2.0)
        self.top_plane = (n_top, p0)

        if self.debug:
            print(f"[DBG] top_plane: n={self._fmt(n_top)}, p0={self._fmt(p0)}")
            self.visualize_walls_and_gripper(
                T_g_base=None, grasp_dims=None,
                pts_base=pts_base,
                approach_dist=None
                )
            print("========== [END BUILD DEBUG] ==========\n")

    @staticmethod
    def _expand_obb(obb, delta):
        new_ext = obb.extent + np.array([delta, delta, delta]) * 2.0
        return o3d.geometry.OrientedBoundingBox(obb.center, obb.R, new_ext)
    
    def _transform_obb_to_base(self, obb_cam: o3d.geometry.OrientedBoundingBox):
        """将相机系的 OBB 变换到基座系：obb_base = T_base_cam ∘ obb_cam"""
        T = self.T_cam_base  # 约定：这是 base_from_cam = T_base_cam
        R_bc = T[:3, :3]
        t_bc = T[:3, 3]
        center_base = (R_bc @ obb_cam.center) + t_bc
        R_base = R_bc @ obb_cam.R
        return o3d.geometry.OrientedBoundingBox(center_base, R_base, obb_cam.extent.copy())

    # ---------- 生成夹爪近似盒（两指+掌），在基座系 ----------
    def _gripper_meshes(self, T_g_base, width, height, depth):
        """
        T_g_base: 4x4 (抓取位姿，与你的 GraspGroup 定义一致)
        width: 抓取开口宽度（指爪内侧距离）
        height: 抓取高度（与 GraspGroup.heights 对齐）
        depth:  抓取深度（与 GraspGroup.depths 对齐）
        """
        meshes = []
        def make_box(size_xyz, T_base):
            mesh = o3d.geometry.TriangleMesh.create_box(*size_xyz)
            mesh.compute_vertex_normals()
            # 平移到中心为原点
            mesh.translate(-np.array(size_xyz)/2.0)
            mesh.transform(T_base)
            return mesh

        # 约定：抓取坐标系 x=接近方向(+x 指向物体)，y=左右，z=高度
        t = self.finger_t; L = self.finger_len; h = float(height); w = float(width)

        # 两指（长度 L，厚度 t，高度 h），中心位于 x = depth - L/2，y = ±(w/2 + t/2)
        T_left  = np.eye(4); T_left[0,3]  = depth - L/2.0; T_left[1,3]  = +(w/2.0 + t/2.0)
        T_right = np.eye(4); T_right[0,3] = depth - L/2.0; T_right[1,3] = -(w/2.0 + t/2.0)
        meshes.append(make_box([L, t, h], T_g_base @ T_left))
        meshes.append(make_box([L, t, h], T_g_base @ T_right))

        # 掌/底部（厚度 t，宽度 w+2t，高度 h），中心位于 x = depth - L - t/2
        T_palm = np.eye(4); T_palm[0,3] = depth - L - t/2.0
        meshes.append(make_box([t, w + 2*t, h], T_g_base @ T_palm))
        return meshes

    @staticmethod
    def _aabb_overlap(mesh, obb, grasp_group, margin: float = 0.0)-> bool:
        # 粗略快速判定：AABB 相交（足够做筛选，后续仍有 MFCD）

        def _points_of(g):
            if hasattr(g, 'to_legacy'):  # o3d.t.* → legacy
                g = g.to_legacy()
            if isinstance(g, o3d.geometry.TriangleMesh):
                return np.asarray(g.vertices) if len(g.vertices) else None
            if isinstance(g, o3d.geometry.PointCloud):
                return np.asarray(g.points)   if len(g.points)   else None
            if isinstance(g, o3d.geometry.LineSet):
                return np.asarray(g.points)   if len(g.points)   else None
            raise TypeError(f'不支持的几何类型：{type(g)}')

        R = np.asarray(obb.R, dtype=np.float64)
        c = np.asarray(obb.center, dtype=np.float64)
        e = 0.5 * np.asarray(obb.extent, dtype=np.float64) + margin  # 只对 slab 进行6面扩张

        P = _points_of(mesh)
        if P is None:
            return False  # 空几何，视为不相交

        Pl = (P - c) @ R                      # (N,3) world→slab-local
        mn = Pl.min(axis=0)                   # mesh 在 slab 局部的最小坐标
        mx = Pl.max(axis=0)                   # mesh 在 slab 局部的最大坐标

        # show_minimal_bbox(aabb_m,aabb_o,grasp_group)

        # —— 三轴区间重叠（闭区间：接触也算）——
        return (mn[0] <= +e[0] and mx[0] >= -e[0]) and \
            (mn[1] <= +e[1] and mx[1] >= -e[1]) and \
            (mn[2] <= +e[2] and mx[2] >= -e[2])

    @staticmethod
    def _point_in_obb(obb: o3d.geometry.OrientedBoundingBox, p, eps: float = 0.0) -> bool:
        """
        兼容 CUDA 的点-OBB 包含测试：True 表示点 p 位于 obb 内（含边界）。
        eps 为可选裕度（米）。
        """
        R = np.asarray(obb.R, dtype=np.float64)
        c = np.asarray(obb.center, dtype=np.float64)
        e = np.asarray(obb.extent, dtype=np.float64) * 0.5  # 半长
        pl = (np.asarray(p, dtype=np.float64) - c) @ R      # world -> local
        return (abs(pl[0]) <= e[0] + eps) and (abs(pl[1]) <= e[1] + eps) and (abs(pl[2]) <= e[2] + eps)

    def _path_intersect_obb(self, T_g, approach_dist):
        """ 预抓取→抓取的路径是否穿过 OBB（抽样 3~4 个点判定） """
        # 约定：沿 -x 退距离得到预抓取（与 GraspNet Baseline 的“shifting”一致）
        num_samp = 4
        for s in np.linspace(0.0, 1.0, num_samp, endpoint=True):
            T = T_g.copy()
            dir_x = T[:3, 0]
            T[:3,3] += -(approach_dist * (1.0 - s)) * dir_x
            p = T[:3, 3]
            if self._point_in_obb(self.obb_forbid, p):
                return True
        return False
    
    def _path_intersect_slabs(self, T_g, approach_dist):
        num_samp = 4
        for s in np.linspace(0.0, 1.0, num_samp, endpoint=True):
            T = T_g.copy()
            dir_x = T[:3, 0]  # 夹爪 x 轴（在当前坐标系下）
            T[:3,3] += -(approach_dist * (1.0 - s)) * dir_x
            p = T[:3, 3]
            if any(self._point_in_obb(slab, p) for slab in self.forbid_slabs):
                return True
        return False

    def _near_rim(self, T_g, width):
        """ 指尖在箱口平面上的投影距离箱沿是否过近（rim_band） """
        if self.top_plane is None:
            return False
        n, p0 = self.top_plane
        # 指尖（两指内侧）在抓取系：y=±w/2, x=depth，z=0
        p_left_g  = np.array([0.0, +width/2.0, 0.0, 1.0])
        p_right_g = np.array([0.0, -width/2.0, 0.0, 1.0])
        P = np.stack([T_g @ p_left_g, T_g @ p_right_g], axis=0)[:, :3]  # 到基座系

        # 投影到箱口平面
        def proj_to_plane(p):
            return p - np.dot(p - p0, n) * n
        Pp = np.stack([proj_to_plane(P[0]), proj_to_plane(P[1])], axis=0)

        # 与 OBB 顶面矩形的边界距离（近似用 OBB 在该平面内的外接矩形）
        # 做法：把点和 OBB 一起变换到 OBB 局部坐标，再看 z=+extent_z/2 平面上的矩形距离
        R, c, e = self.obb.R, self.obb.center, self.obb.extent/2.0
        P_loc = (Pp - c) @ R    # (2,3)
        # 顶面 z ≈ +e[2]; 在 x-y 平面矩形边界的最小外距
        def rect_edge_dist_xy(pxy, ex, ey):
            dx = max(abs(pxy[0]) - ex, 0.0)
            dy = max(abs(pxy[1]) - ey, 0.0)
            return np.hypot(dx, dy)
        d0 = rect_edge_dist_xy(P_loc[0,:2], e[0], e[1])
        d1 = rect_edge_dist_xy(P_loc[1,:2], e[0], e[1])
        return (d0 < self.rim_band) or (d1 < self.rim_band)
    
    def _make_wall_slabs(self, obb, wall_t=0.005, expand=0.0):
        """
        用与 obb 同 R 的 5 个薄板 OBB 近似墙体（允许开口面穿过），再按 expand 膨胀。
        返回：list[OrientedBoundingBox]
        """
        R = obb.R # 世界坐标系下
        c = obb.center # 世界坐标系
        e = obb.extent / 2.0  # OBB尺寸

        # 正交化（数值保护）
        U, _, Vt = np.linalg.svd(R, full_matrices=False)
        R = U @ Vt

        x, y, z = R[:, 0], R[:, 1], R[:, 2] # 局部坐标系到相机坐标系
        slabs = []

        def add_slab(center, ex, ey, ez):
            box = o3d.geometry.OrientedBoundingBox(center, R, np.array([ex, ey, ez]))
            # if expand > 0:
            #     box = o3d.geometry.OrientedBoundingBox(box.center, box.R, box.extent + expand)
            slabs.append(box)

        t = float(wall_t)
        th = max(1e-6, t + 2*expand)  # ❷ 仅法向外凸的总厚度

        # +X 面 / -X 面（侧墙）
        add_slab(c + x*(+e[0]+t/2), th, 2*e[1], 2*e[2])   # +X
        add_slab(c + x*(-e[0]-t/2), th, 2*e[1], 2*e[2])   # -X

        # +Y 面 / -Y 面（侧墙）
        add_slab(c + y*(+e[1]+t/2), 2*e[0], th, 2*e[2])   # +Y
        add_slab(c + y*(-e[1]-t/2), 2*e[0], th, 2*e[2])   # -Y

        # 底面（不开口那一面）
        # 下移底部 
        offset = 0.04
        add_slab(c + z*(-e[2]-t/2-offset), 3*e[0], 3*e[1], th)   # +Z 底

        return slabs

    def visualize_walls_and_gripper(
    self,
    T_g_base: np.ndarray = None,
    grasp_dims: tuple = None,          # (width, height, depth)
    pts_base: np.ndarray = None,       # 可选：基座系点云，仅用于渲染背景
    approach_dist: float = None,       # 可选：若给出，则画接近路径
    path_samples: int = 6,             # 接近路径采样数
    show_obb_wire: bool = True,        # 线框 OBB
    show_slabs_mesh: bool = True,      # 以实体薄盒显示墙体（更直观）
    gg=None
):
        """
        可视化：墙体（slabs）、抓取手型（两指+掌）、可选点云与接近路径、原始手型。
        依赖：self.obb, self.forbid_slabs, self._gripper_meshes(...)
        """
        geoms = []

        # 背景点云（可选）
        if pts_base is not None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_base.astype(np.float64)))
            pcd.paint_uniform_color([0.55, 0.55, 0.65])
            geoms.append(pcd)

        # OBB 线框
        if show_obb_wire and (self.obb is not None):
            ls_obb = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.obb)
            ls_obb.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(ls_obb)

        # 墙体（slabs）：用实体薄盒展示（可选），否则线框
        if self.forbid_slabs is not None:
            for slab in self.forbid_slabs:
                if show_slabs_mesh:
                    geoms.append(_obb_to_mesh(slab, color=(1.0, 0.6, 0.0)))
                else:
                    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(slab)
                    ls.paint_uniform_color([1.0, 0.6, 0.0])
                    geoms.append(ls)

        # 抓取手型网格（三盒体）
        path_ls = None
        if (T_g_base is not None) and (grasp_dims is not None):
            w, h, d = map(float, grasp_dims)
            meshes = self._gripper_meshes(T_g_base, w, h, d)
            # 分别着色：左/右指、掌
            if len(meshes) == 3:
                meshes[0].paint_uniform_color([0.2, 0.4, 1.0])  # left finger
                meshes[1].paint_uniform_color([0.2, 0.4, 1.0])  # right finger
                meshes[2].paint_uniform_color([0.9, 0.2, 0.2])  # palm
            geoms.extend(meshes)

            # 可选：接近路径折线
            if (approach_dist is not None) and (approach_dist > 0) and (path_samples >= 2):
                dir_x = T_g_base[:3, 0]
                dir_x /= (np.linalg.norm(dir_x) + 1e-12)
                pts = []
                for s in np.linspace(0.0, 1.0, path_samples, endpoint=True):
                    Ttmp = T_g_base.copy()
                    Ttmp[:3, 3] += -(approach_dist * (1.0 - s)) * dir_x
                    pts.append(Ttmp[:3, 3].copy())
                pts = np.asarray(pts, dtype=np.float64)
                path_ls = _make_path_lines(pts, color=(0.2, 0.8, 1.0))
                if path_ls is not None:
                    geoms.append(path_ls)

        # 坐标系
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        # 原始手型
        if gg != None:
            grippers = gg.to_open3d_geometry_list()
            geoms.extend([*grippers])

        # 展示
        o3d.visualization.draw_geometries(geoms)

    def visualize_walls_and_gripper_batch(
        self,
        T_g_base: np.ndarray = None,
        grasp_dims: tuple = None,          # (width, height, depth) 或 (N,3)
        pts_base: np.ndarray = None,       # 可选：基座系点云，仅用于渲染背景
        approach_dist: float = None,       # 可选：若给出，则画接近路径
        path_samples: int = 6,             # 接近路径采样数
        show_obb_wire: bool = True,        # 线框 OBB
        show_slabs_mesh: bool = True       # 以实体薄盒显示墙体（更直观）
    ):
        """
        可视化：墙体（slabs）、抓取手型（两指+掌）、可选点云与接近路径。
        依赖：self.obb, self.forbid_slabs, self._gripper_meshes(...)
        说明：T_g_base 与 grasp_dims 既可为单个，也可为批量（N 个）。
        """
        geoms = []

        # 背景点云（可选）
        if pts_base is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_base.astype(np.float64))
            pcd.paint_uniform_color([0.55, 0.55, 0.65])
            geoms.append(pcd)

        # OBB 线框
        if show_obb_wire and (self.obb is not None):
            ls_obb = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.obb)
            ls_obb.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(ls_obb)

        # 墙体（slabs）：用实体薄盒展示（可选），否则线框
        if self.forbid_slabs is not None:
            for slab in self.forbid_slabs:
                if show_slabs_mesh:
                    geoms.append(_obb_to_mesh(slab, color=(1.0, 0.6, 0.0)))
                else:
                    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(slab)
                    ls.paint_uniform_color([1.0, 0.6, 0.0])
                    geoms.append(ls)

        # =========================
        # 抓取手型网格（三盒体）—支持批量
        # =========================
        if (T_g_base is not None) and (grasp_dims is not None):
            T_arr = np.asarray(T_g_base)
            D_arr = np.asarray(grasp_dims, dtype=np.float64)

            # 兼容单个输入：统一转为批量第一维 N
            if T_arr.ndim == 2 and T_arr.shape == (4, 4):
                T_arr = T_arr[None, ...]          # -> (1,4,4)
            if D_arr.ndim == 1 and D_arr.size == 3:
                D_arr = D_arr[None, ...]          # -> (1,3)

            # 若只给了一个尺寸而有多个位姿，则广播尺寸
            if D_arr.shape[0] == 1 and T_arr.shape[0] > 1:
                D_arr = np.repeat(D_arr, T_arr.shape[0], axis=0)

            assert T_arr.shape[0] == D_arr.shape[0], \
                f"N(T_g_base)={T_arr.shape[0]} 与 N(grasp_dims)={D_arr.shape[0]} 不一致"

            # 批量生成几何
            for k in range(T_arr.shape[0]):
                w, h, d = map(float, D_arr[k])
                T_k = T_arr[k]

                meshes = self._gripper_meshes(T_k, w, h, d)
                # 分别着色：左/右指、掌（保持你原有配色）
                if len(meshes) == 3:
                    meshes[0].paint_uniform_color([0.2, 0.4, 1.0])  # left finger
                    meshes[1].paint_uniform_color([0.2, 0.4, 1.0])  # right finger
                    meshes[2].paint_uniform_color([0.9, 0.2, 0.2])  # palm
                geoms.extend(meshes)

                # 可选：接近路径折线（逐个抓取都画）
                if (approach_dist is not None) and (approach_dist > 0) and (path_samples >= 2):
                    dir_x = T_k[:3, 0]
                    nrm = np.linalg.norm(dir_x) + 1e-12
                    dir_x = dir_x / nrm
                    s_vals = np.linspace(0.0, 1.0, path_samples, endpoint=True)
                    P = np.stack([
                        T_k[:3, 3] - (approach_dist * (1.0 - s)) * dir_x
                        for s in s_vals
                    ], axis=0)
                    geoms.append(_make_path_lines(P, color=(0.2, 0.8, 1.0)))

        # 坐标系
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

        # 展示（只打开一次窗口）
        o3d.visualization.draw_geometries(geoms)

    # ---------- 对一批抓取做过滤 ----------
    def filter_grasps(self, grasp_group, approach_dist=0.05,pts_cam=None):
        if self.obb_forbid is None:
            raise RuntimeError("请先调用 build(mask_box, depth_m) 完成 OBB 构建。")
        # 下面的T、R是相对啥的坐标系？，搞清楚！！和点云坐标好像是一个，即为相机系下
        T = grasp_group.translations        # (M,3)
        R = grasp_group.rotation_matrices   # (M,3,3)
        H = grasp_group.heights             # (M,)
        D = grasp_group.depths              # (M,)
        W = grasp_group.widths              # (M,)

        M = T.shape[0]
        forbid = np.zeros((M,), dtype=bool)

        pts_base = (to_homo(pts_cam) @ self.T_cam_base.T)[:, :3]
        # ------------准备逆变换和缓存 ------------
        T_base_cam = np.linalg.inv(self.T_cam_base)
        new_R_cam, new_t_cam = [], []

        T_list = []
        dims_list = []
        for i in range(M):
            # 构造抓取位姿（4x4），与 GraspNet Baseline 坐标约定保持一致
            # R, T 来自 GraspGroup：
            # - R: camera -> gripper
            # - T: gripper 原点在 camera 系的位置
            R_c2g = R[i]               # (3,3)
            t_cam = T[i]               # (3,)

            # 1) gripper载camera下的位姿
            T_g_cam = np.eye(4)
            T_g_cam[:3, :3] = R_c2g
            T_g_cam[:3,  3] = t_cam

            # 2) 相机 -> 基座，把夹爪放到 "base" 系
            T_g_base = self.T_cam_base @ T_g_cam

            if self.aling_grasp:
                # ========= 新增：把进给方向(夹爪x轴)对齐到底板法向，四元数版 =========
                # 1) 准备：底板法向(基座坐标系下)。建议在 build() 里放到 self.bottom_normal_base
                n_base = getattr(self, "bottom_normal_base", np.array([0.0, 0.0, 1.0], dtype=float))
                n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

                # 2) 当前抓取在基座系下的 x 轴（进给方向）
                g_x_base = T_g_base[:3, 0]               # 列向量就是基座下的夹爪局部轴
                g_x_base = g_x_base / (np.linalg.norm(g_x_base) + 1e-12)

                # 3) 为了最小转角：选择与当前 g_x_base 最近的 ±n_base 作为目标方向
                target = n_base if np.dot(g_x_base, n_base) >= 0 else -n_base

                # 4) 用四元数算“从 g_x_base 旋到 target”的校正旋转（左乘，作用在基座系）
                q_b2g    = mat_to_quat(T_g_base[:3, :3])             # 基座->夹爪 的当前姿态（四元数）
                q_align  = quat_from_two_vectors(g_x_base, target)   # 让 x 轴对齐的最小转角四元数
                q_new    = quat_mul(q_align, q_b2g)                  # 先对齐(基座系)，再到夹爪

                # 5) 回写旋转矩阵（平移不变）
                T_g_base[:3, :3] = quat_to_mat(q_new)

                # （可选）若你后面还想用回 camera 系的 R[i]，可回推回去以保持一致：
                # R_new_c2g = self.T_cam_base[:3, :3].T @ T_g_base[:3, :3]
                # R[i] = R_new_c2g
                dir_x = T_g_base[:3, 0]
                n    = self.bottom_normal_base
                target = n if np.dot(dir_x, n) >= 0 else -n       # 与 ±n 中夹角更小的那个
                R_align = _rot_between_vecs_quat(dir_x, target)   # 四元数求旋转
                T_g_base[:3, :3] = R_align @ T_g_base[:3, :3]
                # ========= 新增结束 =========

            # (1) 指爪/掌 与 膨胀后的箱体 OBB 粗碰撞
            meshes = self._gripper_meshes(T_g_base, W[i], H[i], D[i])

            if self.debug:
                T_list.append(T_g_base)
                dims_list.append((W[i], H[i], D[i]))
                
                test = grasp_group.to_open3d_geometry_list()
                test.extend(meshes)
                o3d.visualization.draw_geometries(test)

            if self.debug and i < self.max_print_grasps:
                print(f"\n----- [FILTER DEBUG] grasp {i} -----")
                self._print_frame_stats("T_g_cam (R|t)", T_g_cam[:3,:3], T_g_cam[:3,3])
                self._print_frame_stats("T_g_base (R|t)", T_g_base[:3,:3], T_g_base[:3,3])
                # 打印第一块 mesh 与第一块 slab AABB
                self._print_aabb("mesh0(base)", meshes[0])
                if len(self.forbid_slabs) > 0:
                    self._print_aabb("slab0(base)", self.forbid_slabs[0])
                # 采样接近路径点，查看是否命中 slab
                num_samp = 4
                pts_trace = []
                for s in np.linspace(0.0, 1.0, num_samp, endpoint=True):
                    Ttmp = T_g_base.copy()
                    dir_x = T_g_base[:3, 0]  # 夹爪 x 轴在基座系下的方向
                    Ttmp[:3,3] += -(approach_dist * (1.0 - s)) * dir_x
                    pts_trace.append(Ttmp[:3,3].copy())
                pts_trace = np.array(pts_trace)
                hit_any = any(self._point_in_obb(slab, p) for p in pts_trace for slab in self.forbid_slabs)
                
                self.visualize_walls_and_gripper(
                        T_g_base=T_g_base,
                        grasp_dims=(W[i], H[i], D[i]),
                        pts_base=pts_base,                # 如需显示背景点云可传入
                        approach_dist=approach_dist,  # 画接近路径
                        path_samples=6,
                        show_obb_wire=True,
                        show_slabs_mesh=True,
                        gg=grasp_group
                    )
                print(f"[DBG] approach_trace points (base):\n{self._fmt(pts_trace)}")
                print(f"[DBG] approach_trace hit_any_slab = {hit_any}")

            # (1) AABB 粗筛： slab的aabb和夹爪meshes的aabb相交才 forbid
            if any(self._aabb_overlap(m, slab, grasp_group, margin=5e-3)  # 1mm 裕度可调
                for m in meshes for slab in self.forbid_slabs):
                forbid[i] = True
                continue

            # # (2) 夹爪前进方向
            # if self._path_intersect_slabs(T_g_base, approach_dist=approach_dist):
            #     forbid[i] = True
            #     continue

            # # (3) 距箱沿过近
            # if self._near_rim(T_g_base, W[i]):
            #     forbid[i] = True
            #     continue
        
            T_g_cam_new = T_base_cam @ T_g_base
            new_R_cam.append(T_g_cam_new[:3, :3].copy())
            new_t_cam.append(T_g_cam_new[:3,  3].copy())

        if self.debug:
            # 2) 拼成批量数组
            T_batch   = np.stack(T_list, axis=0)          # (M,4,4)
            dims_batch= np.asarray(dims_list, dtype=float)  # (M,3)

            # 3) 一次性调用，统一渲染
            # 若需要背景点云：
            # pts_base = (to_homo(pts_cam) @ self.T_cam_base.T)[:, :3]
            self.visualize_walls_and_gripper_batch(
                T_g_base=T_batch,
                grasp_dims=dims_batch,
                pts_base=pts_base if 'pts_base' in locals() else None,
                approach_dist=approach_dist,
                path_samples=6,
                show_obb_wire=True,
                show_slabs_mesh=True
            )

            new_R_cam = np.stack(new_R_cam, axis=0)
        
        if len(new_R_cam) == 0:
            # 全部被过滤：返回空的 GraspGroup 子集
            empty_idx = np.array([], dtype=int)
            return forbid, grasp_group[empty_idx]
        new_R_cam = np.stack(new_R_cam, axis=0)   # (M_valid, 3, 3)
        new_t_cam = np.stack(new_t_cam, axis=0)   # (M_valid, 3)
        #计算有效索引，并把 grasp_group 缩小为同样的子集
        valid_idx = np.flatnonzero(~forbid)       # 长度 = M_valid
        grasp_group = grasp_group[valid_idx]
        grasp_group.rotation_matrices = new_R_cam
        grasp_group.translations      = new_t_cam

        return forbid, grasp_group

def show_minimal_bbox(geom, obb,grasp_group):
    """
    简单的可视化：
    - geom: open3d TriangleMesh / PointCloud / 或 Nx3 numpy 点
    - obb:  open3d OrientedBoundingBox 或 AxisAlignedBoundingBox
    """
    # 统一成 Open3D 几何
    if isinstance(geom, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(geom.astype(np.float32))
        geom = pcd
            # 原始手型

    # 本体 AABB（红色）
    aabb = geom.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    # 输入的 OBB/AABB（绿色）
    if not isinstance(obb, (o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox)):
        raise TypeError("obb 必须是 open3d 的 OrientedBoundingBox 或 AxisAlignedBoundingBox")
    obb.color = (0, 1, 0)

    if grasp_group != None:
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([geom, aabb, obb, *grippers], window_name="Minimal BBox Viz")

    else:
        o3d.visualization.draw_geometries([geom, aabb, obb], window_name="Minimal BBox Viz")