# python scripts/bvh_to_robot.py \
#   --bvh_file "/mnt/c/Users/TsingSV/Work/GMR/test_Skeleton0.bvh" \
#   --format lafan1 \
#   --robot unitree_g1 \
#   --motion_fps 120 \
#   --rate_limit \
#   --save_path "/mnt/c/Users/TsingSV/Work/GMR/test_Skeleton0_g1.pkl" \
#   --record_video \
#   --video_path /mnt/c/Users/TsingSV/Work/GMR/test_Skeleton0_g1.mp4


# python scripts/bvh_to_robot.py \
#   --bvh_file /mnt/c/Users/TsingSV/GMR/aiming1_subject1.bvh \
#   --format lafan1 \
#   --robot unitree_g1 \
#   --motion_fps 120 \
#   --rate_limit \
#   --save_path /mnt/c/Users/TsingSV/GMR/lafan1.pkl


python scripts/bvh_to_robot.py \
  --bvh_file /mnt/c/Users/TsingSV/GMR/Skeleton1.bvh \
  --format mocap \
  --robot unitree_g1 \
  --motion_fps 120 \
  --rate_limit \
  --save_path /mnt/c/Users/TsingSV/GMR/skeleton1.pkl
  #  \
  # --record_video \
  # --video_path /mnt/c/Users/TsingSV/GMR/test_Skeleton0_g1.mp4


# python scripts/vis_robot_motion.py \
#   --robot unitree_g1 \
#   --robot_motion_path /mnt/c/Users/TsingSV/Work/GMR/test_Skeleton0_g1.pkl
