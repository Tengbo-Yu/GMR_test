# 将实时传输的机器人关节角度发送到对应ip

python scripts/send_lumo_g1_robot_array.py \
  --ip 192.168.50.150 \
  --robot-name UnitreeG1 \
  --dst-ip 192.168.123.164 \
  --dst-port 9000 \
  --protocol udp \
  --joint-smoothing-alpha 1.0 \
  --visualize \
  --motion-fps 120 \
  --rate-limit \
  --non-blocking \
  --print-joint-summary
