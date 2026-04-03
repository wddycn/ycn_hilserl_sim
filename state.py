import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def numpy_to_list(stats):
    out = {}
    for k, v in stats.items():
        if k in ["min", "max", "mean", "std"]:
            out[k] = v.tolist()
    return out

def main():
    repo_id = "wddycn/hilserl_sim_pick_lift"
    root = Path("/home/ycn/hilserl_sim/lerobot/src/lerobot/ycn_hilserl_sim/ycn_records_gamepad")  

    ds = LeRobotDataset(repo_id=repo_id, root=root)

    # 这一步里 meta.stats 就是 aggregate_stats 之后的结果
    all_stats = ds.meta.stats   # dict: feature_name -> {min, max, mean, std, ...}

    # 转成 JSON 可写格式（numpy -> list）
    json_stats = {}
    for feat_name, feat_stats in all_stats.items():
        json_stats[feat_name] = numpy_to_list(feat_stats)

    # 输出到文件
    out_path = Path("/home/ycn/hilserl_sim/lerobot/src/lerobot/ycn_hilserl_sim/dataset_stats.json")
    out_path.write_text(json.dumps(json_stats, indent=4))
    print(f"saved to {out_path}")

if __name__ == "__main__":
    main()
