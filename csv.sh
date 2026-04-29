# local_csv.sh 回放 LuMo CSV 到 remote.sh
python scripts/send_lumo_csv_human_stream.py \
  --csv_path /mnt/c/Users/TsingSV/GMR/misc/0428.csv \
  --dst_ip 192.168.50.34 \
  --dst_port 19000 \
  --protocol udp \
  --print_joint_summary
