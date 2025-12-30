
COMMAND="python main.py --device cuda:0 --log "

DATA=$1

N=(1000 800 600 400 200 100)

if [ "$DATA" = "mnist" ]; then
  arch="cnn8"
elif [ "$DATA" = "cifar" ]; then
  arch="vgg13"
elif [ "$DATA" = "svhn" ]; then
  arch="vgg13"
elif [ "$DATA" = "gtsrb" ]; then
  arch="vgg11"
elif [ "$DATA" = "imagentte" ]; then
  arch="vgg16"
else
  echo "Unsupported dataset: $DATA"
  exit 1
fi

ATTACKS=("Badnets" "Blend")
for ATT in "${ATTACKS[@]}"; do
  for n in "${N[@]}"; do
    FULL_COMMAND="$COMMAND --dataset $DATA --attack $ATT --N $n --N_clean $n"
    echo "Running: $FULL_COMMAND"
    $FULL_COMMAND
  done
done

