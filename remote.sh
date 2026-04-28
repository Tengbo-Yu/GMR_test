python scripts/receive_human_stream_retarget.py \
  --listen_host 0.0.0.0 \
  --listen_port 19000 \
  --protocol udp \
  --robot unitree_g1 \
  --motion_fps 120 \
  --rate_limit \
  --forward_udp_port 9000


# python scripts/receive_human_stream_retarget.py \
#   --listen_host 0.0.0.0 \
#   --listen_port 19000 \
#   --protocol udp \
#   --dry_run
