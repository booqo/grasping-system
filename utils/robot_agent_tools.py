# robot_agent_tools.py
from dataclasses import dataclass, field
from typing import Union, Dict, Any, List
import numpy as np
import cv2
import base64
from qwen_agent.tools.base import BaseTool, register_tool

# 运行期句柄存储（避免在消息里传大数据）

@dataclass
class RuntimeStore:
    obs: dict = field(default_factory=dict)      # obs_id -> {"color", "depth", "mask"}
    pcd: dict = field(default_factory=dict)      # pc_id  -> {"end_points", "cloud_o3d"}
    grasp: dict = field(default_factory=dict)    # grasp_id-> {"gg"}
    _obs_cnt: int = 0
    _pc_cnt: int = 0
    _gr_cnt: int = 0
    def new_obs_id(self) -> str: self._obs_cnt += 1; return f"obs_{self._obs_cnt}"
    def new_pc_id(self) -> str:  self._pc_cnt += 1;  return f"pc_{self._pc_cnt}"
    def new_grasp_id(self) -> str:self._gr_cnt += 1; return f"grasp_{self._gr_cnt}"

# ---------------- 感知工具 ----------------
@register_tool("perception_use")
class PerceptionUse(BaseTool):
    """
    渲染RGB/Depth -> 分割 -> 点云构建。
    依赖 cfg: env, store, segment_image, get_and_process_data
    """
    parameters = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {"type":"string","enum":["snapshot","segment","pointcloud"]},
            "obs_id": {"type":"string","description":"segment/pointcloud 需要"}
        }
    }
    def __init__(self, cfg=None):
        self.env = cfg["env"]
        self.store: RuntimeStore = cfg["store"]
        self.segment_image = cfg["segment_image"]
        self.get_and_process_data = cfg["get_and_process_data"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        p = self._verify_json_format_args(params)
        a = p["action"]
        try:
            if a == "snapshot":   return self._snapshot()
            if a == "segment":    return self._segment(p["obs_id"])
            if a == "pointcloud": return self._pointcloud(p["obs_id"])
            return {"status":"failure","message":f"unknown action={a}"}
        except Exception as e:
            return {"status":"failure","message":f"perception_use error: {e}"}

    def _snapshot(self) -> Dict[str, Any]:
        imgs = self.env.render()   # {"img": RGB(HxWx3, uint8), "depth": float32 HxW}
        color, depth = imgs["img"], imgs["depth"]
        obs_id = self.store.new_obs_id()
        self.store.obs[obs_id] = {"color": color, "depth": depth, "mask": None}
        # 生成缩略图
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", bgr)
        preview = base64.b64encode(jpg.tobytes()).decode("utf-8") if ok else ""
        return {"status":"success","obs_id":obs_id,"preview_jpeg_base64":preview}

    def _segment(self, obs_id: str) -> Dict[str, Any]:
        rec = self.store.obs.get(obs_id)
        if rec is None: return {"status":"failure","message":f"obs_id={obs_id} not found"}
        mask = self.segment_image(rec["color"])
        self.store.obs[obs_id]["mask"] = mask
        return {"status":"success","obs_id":obs_id}

    def _pointcloud(self, obs_id: str) -> Dict[str, Any]:
        rec = self.store.obs.get(obs_id)
        if rec is None: return {"status":"failure","message":f"obs_id={obs_id} not found"}
        if rec["mask"] is None:
            return {"status":"failure","message":"call segment first (mask is None)."}
        end_points, cloud_o3d = self.get_and_process_data(rec["color"], rec["depth"], rec["mask"])
        pc_id = self.store.new_pc_id()
        self.store.pcd[pc_id] = {"end_points": end_points, "cloud_o3d": cloud_o3d}
        npts = len(np.asarray(cloud_o3d.points))
        return {"status":"success","pc_id":pc_id,"num_points":int(npts)}

# ---------------- 抓取规划工具 ----------------
@register_tool("grasp_plan_use")
class GraspPlanUse(BaseTool):
    """
    基于点云进行抓取规划，返回 GraspGroup 句柄。
    依赖 cfg: store, generate_grasps
    """
    parameters = {
        "type": "object",
        "required": ["action","pc_id"],
        "properties": {
            "action": {"type":"string","enum":["plan_grasp"]},
            "pc_id": {"type":"string"},
            "visual": {"type":"boolean","default": False}
        }
    }
    def __init__(self, cfg=None):
        self.store: RuntimeStore = cfg["store"]
        self.generate_grasps = cfg["generate_grasps"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        p = self._verify_json_format_args(params)
        if p["action"] != "plan_grasp":
            return {"status":"failure","message":"unknown action"}
        try:
            data = self.store.pcd.get(p["pc_id"])
            if data is None: return {"status":"failure","message":f"pc_id={p['pc_id']} not found"}
            gg = self.generate_grasps(data["end_points"], data["cloud_o3d"], bool(p.get("visual", False)))
            grasp_id = self.store.new_grasp_id()
            self.store.grasp[grasp_id] = {"gg": gg}
            best = gg[0]
            return {"status":"success","grasp_id":grasp_id,
                    "best_grasp":{"translation":best.translation.tolist(),
                                  "rotation_matrix":best.rotation_matrix.tolist(),
                                  "width": float(best.width),
                                  "score": float(best.score)}}
        except Exception as e:
            return {"status":"failure","message":f"plan error: {e}"}

# ---------------- 运动执行工具 ----------------
@register_tool("motion_use")
class MotionUse(BaseTool):
    """
    直接调用你现有的 execute_grasp(env, gg) 完成一次抓取流程。
    依赖 cfg: env, store, execute_grasp
    """
    parameters = {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {"type":"string","enum":["exec_grasp","wait","terminate"]},
            "grasp_id": {"type":"string"},
            "time": {"type":"number"},
            "status": {"type":"string","enum":["success","failure"]}
        }
    }
    def __init__(self, cfg=None):
        self.env = cfg["env"]
        self.store: RuntimeStore = cfg["store"]
        self.execute_grasp = cfg["execute_grasp"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        p = self._verify_json_format_args(params)
        a = p["action"]
        try:
            if a == "exec_grasp":
                gid = p.get("grasp_id")
                rec = self.store.grasp.get(gid)
                if rec is None: return {"status":"failure","message":f"grasp_id={gid} not found"}
                self.execute_grasp(self.env, rec["gg"])
                return {"status":"success","message":"grasp executed"}
            if a == "wait":
                import time; time.sleep(max(0.0, float(p.get("time", 0.0))))
                return {"status":"success","message":f"wait {p.get('time',0)}s"}
            if a == "terminate":
                return {"status": p.get("status","success"), "message":"terminated"}
            return {"status":"failure","message":f"unknown action={a}"}
        except Exception as e:
            return {"status":"failure","message":f"exec error: {e}"}

# ---------------- 回0工具 ----------------
# @register_tool("perception_use")
# class PerceptionUse(BaseTool):
#     ...