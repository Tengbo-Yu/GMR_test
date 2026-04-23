import argparse
import os
import pickle
import re
import signal
import xml.etree.ElementTree as ET

import LuMoSDKClient

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "Saving a GMR-style pkl requires numpy. Please install numpy in the Python "
        "environment used to run LuMoSDKPy/PythonSample_export_pkl.py."
    ) from exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_IP = "192.168.50.150"
DEFAULT_SAVE_PATH = os.path.join(REPO_ROOT, "lumo_motor_capture.pkl")
DEFAULT_ROBOT_XML = os.path.join(REPO_ROOT, "assets", "unitree_g1", "g1_mocap_29dof.xml")
DEFAULT_FPS = 120

RUNNING = True


def normalize_joint_name(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_joint_order(robot_xml_path):
    tree = ET.parse(robot_xml_path)
    joint_order = []
    for motor in tree.findall(".//actuator/motor"):
        motor_name = motor.get("name")
        if motor_name:
            joint_order.append(motor_name)
    if not joint_order:
        raise RuntimeError(f"Failed to load actuator order from {robot_xml_path}")
    return joint_order


def build_alias_to_joint_name(joint_order):
    alias_to_joint_name = {}
    manual_aliases = {
        "left_knee_joint": ["left_knee", "left_knee_pitch", "left_knee_pitch_joint"],
        "right_knee_joint": ["right_knee", "right_knee_pitch", "right_knee_pitch_joint"],
        "left_elbow_joint": ["left_elbow", "left_elbow_pitch", "left_elbow_pitch_joint"],
        "right_elbow_joint": ["right_elbow", "right_elbow_pitch", "right_elbow_pitch_joint"],
    }

    for joint_name in joint_order:
        aliases = {joint_name}
        if joint_name.endswith("_joint"):
            aliases.add(joint_name[: -len("_joint")])
        aliases.update(manual_aliases.get(joint_name, []))
        for alias in aliases:
            alias_to_joint_name[normalize_joint_name(alias)] = joint_name

    return alias_to_joint_name


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quat_wxyz(quat):
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def transform_lumo_position(x, y, z):
    return np.array([x, -z, y], dtype=np.float64) * 0.01


def transform_lumo_quat_to_xyzw(qw, qx, qy, qz):
    lumo_to_gmr_wxyz = np.array([0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float64)
    quat_wxyz = normalize_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float64))
    transformed_wxyz = normalize_quat_wxyz(quat_mul_wxyz(lumo_to_gmr_wxyz, quat_wxyz))
    return transformed_wxyz[[1, 2, 3, 0]]


def find_root_bone(skeleton):
    bones_by_name = {bone.Name: bone for bone in skeleton.skeletonBones}
    for candidate in ("Hips", "Pelvis", "pelvis", "Root", "Hip"):
        if candidate in bones_by_name:
            return bones_by_name[candidate]
    if skeleton.skeletonBones:
        return skeleton.skeletonBones[0]
    return None


def select_tracked_skeleton(frame):
    for skeleton in frame.skeletons:
        if skeleton.IsTrack:
            return skeleton
    return None


def build_motion_frame(skeleton, joint_order, alias_to_joint_name):
    raw_motor_angles = {str(key): float(value) for key, value in skeleton.MotorAngle.items()}
    resolved_motor_angles = {}
    unknown_motor_names = []

    for raw_name, value in raw_motor_angles.items():
        canonical_name = alias_to_joint_name.get(normalize_joint_name(raw_name))
        if canonical_name is None:
            unknown_motor_names.append(raw_name)
            continue
        resolved_motor_angles[canonical_name] = value

    dof_pos = np.zeros(len(joint_order), dtype=np.float64)
    missing_joint_names = []
    for index, joint_name in enumerate(joint_order):
        if joint_name in resolved_motor_angles:
            dof_pos[index] = resolved_motor_angles[joint_name]
        else:
            missing_joint_names.append(joint_name)

    root_pos = np.zeros(3, dtype=np.float64)
    root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    root_bone = find_root_bone(skeleton)
    if root_bone is not None:
        root_pos = transform_lumo_position(root_bone.X, root_bone.Y, root_bone.Z)
        root_rot = transform_lumo_quat_to_xyzw(root_bone.qw, root_bone.qx, root_bone.qy, root_bone.qz)

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "missing_joint_names": missing_joint_names,
        "unknown_motor_names": unknown_motor_names,
    }


def save_motion_pkl(motion_frames, save_path, fps):
    if not motion_frames:
        print("No tracked skeleton frames captured, skip pkl saving.")
        return

    motion_data = {
        "fps": fps,
        "root_pos": np.stack([frame["root_pos"] for frame in motion_frames], axis=0),
        "root_rot": np.stack([frame["root_rot"] for frame in motion_frames], axis=0),
        "dof_pos": np.stack([frame["dof_pos"] for frame in motion_frames], axis=0),
        "local_body_pos": None,
        "link_body_list": None,
    }

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(motion_frames)} frames to {save_path}")


def handle_signal(signum, frame):
    del frame
    global RUNNING
    print(f"Received signal {signum}, stopping capture...")
    RUNNING = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture LuMo skeleton MotorAngle data and save a GMR-style pkl."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="LuMo server IP.")
    parser.add_argument("--save_path", default=DEFAULT_SAVE_PATH, help="Output pkl path.")
    parser.add_argument("--robot_xml", default=DEFAULT_ROBOT_XML, help="MuJoCo robot xml for joint order.")
    parser.add_argument("--motion_fps", type=int, default=DEFAULT_FPS, help="fps field written into the pkl.")
    parser.add_argument(
        "--quiet_motor_print",
        action="store_true",
        help="Do not print every MotorAngle value for every frame.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    joint_order = load_joint_order(args.robot_xml)
    alias_to_joint_name = build_alias_to_joint_name(joint_order)
    motion_frames = []
    printed_summary = False

    print(f"Capture target pkl: {args.save_path}")
    print(f"Loaded {len(joint_order)} joints from: {args.robot_xml}")

    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)

    try:
        while RUNNING:
            frame = LuMoSDKClient.ReceiveData(0)
            if frame is None:
                continue

            skeleton = select_tracked_skeleton(frame)
            if skeleton is None:
                continue

            motion_frame = build_motion_frame(skeleton, joint_order, alias_to_joint_name)
            motion_frames.append(motion_frame)

            print(f"Captured frame #{len(motion_frames)} | RobotName: {skeleton.RobotName}")

            if not printed_summary:
                if motion_frame["missing_joint_names"]:
                    print("Missing G1 joints in current MotorAngle data:")
                    for joint_name in motion_frame["missing_joint_names"]:
                        print(joint_name)
                else:
                    print("All G1 joints were found in MotorAngle data.")

                if motion_frame["unknown_motor_names"]:
                    print("MotorAngle names that could not be mapped:")
                    for joint_name in motion_frame["unknown_motor_names"]:
                        print(joint_name)

                printed_summary = True

            if not args.quiet_motor_print:
                for key in skeleton.MotorAngle:
                    print(key)
                    print(skeleton.MotorAngle[key])

    finally:
        LuMoSDKClient.Close()
        save_motion_pkl(motion_frames, args.save_path, args.motion_fps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
