# 本地接收数据＋gmr
python scripts/lumo_live_streaming.py \
  --ip 192.168.50.150 \
  --robot unitree_g1 \
  --motion_fps 120 \
  --rate_limit \
  --template_bvh /mnt/c/Users/TsingSV/GMR/test_Skeleton0.bvh \
  --save_bvh /mnt/c/Users/TsingSV/GMR/lumo_live.bvh \
  --save_path /mnt/c/Users/TsingSV/GMR/lumo_live_g1.pkl \
  --forward_host 192.168.50.34 \
  --forward_port 9000 \
  --forward_protocol udp \
  --forward_format array \
  --non_blocking \
  --record_video \
  --video_path /mnt/c/Users/TsingSV/GMR/lumo_live_g1.mp4 \
  --print_joint_summary

# python scripts/lumo_live_streaming.py \
#   --ip 192.168.50.150 \
#   --robot unitree_g1 \
#   --motion_fps 120 \
#   --rate_limit \
#   --template_bvh /mnt/c/Users/TsingSV/GMR/test_Skeleton0.bvh \
#   --save_bvh /mnt/c/Users/TsingSV/GMR/lumo_live.bvh \
#   --save_path /mnt/c/Users/TsingSV/GMR/lumo_live_g1.pkl \
#   --forward_host 127.0.0.1 \
#   --forward_port 9000 \
#   --forward_protocol udp \
#   --forward_format pickle \
#   --non_blocking \
#   --record_video \
#   --video_path /mnt/c/Users/TsingSV/GMR/lumo_live_g1.mp4
