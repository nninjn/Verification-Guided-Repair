#!/bin/bash

COMMAND_BASE="python main.py --log --device cuda:0"

# 1. Check/Input for N_cor ($1)
if [ -z "$1" ]; then
  read -p "Please enter the value for N_cor (e.g., 500): " N_COR
else
  N_COR=$1
fi

# 2. Check/Input for N_vio ($2)
if [ -z "$2" ]; then
  read -p "Please enter the value for N_vio (e.g., 500): " N_VIO
else
  N_VIO=$2
fi

declare -a all_commands=()

for x in 2 3 4 5; do
  for y in {1..9}; do
    model="n${x}${y}"

    if [[ "$model" != "n33" && "$model" != "n42" ]]; then
      
      FULL_COMMAND_P2="$COMMAND_BASE --model $model --N_cor $N_COR --N_vio $N_VIO --property p2"
      all_commands+=("$FULL_COMMAND_P2")
      
      if [[ "$model" == "n29" ]]; then
        FULL_COMMAND_P8="$COMMAND_BASE --model $model --N_cor $N_COR --N_vio $N_VIO --property p8"
        all_commands+=("$FULL_COMMAND_P8")
      fi
    fi
  done
done

FULL_COMMAND_N19="$COMMAND_BASE --model n19 --N_cor $N_COR --N_vio $N_VIO --property p7"
all_commands+=("$FULL_COMMAND_N19")

echo "⚙️ All ${#all_commands[@]} repair tasks..."
echo "---"
echo "N_cor set to: $N_COR"
echo "N_vio set to: $N_VIO"
echo "---"

for cmd in "${all_commands[@]}"; do
  
  # 输出并执行
  echo "Running: $cmd"
  $cmd
  
done

echo "---"
echo "✅ Repair done."