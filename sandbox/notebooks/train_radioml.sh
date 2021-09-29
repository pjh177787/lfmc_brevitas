PYTHON="/home/aperture/anaconda3/envs/brevitas/bin/python"

DATE=`date +%Y-%m-%d`

# models: 
# 'binput-prerprelu-pg', 
# 'binput-prerprelu-1bit',
# 'binput-prerprelu-2bit', 
# 'prerprelu-resnet20'
arch=UltraNet
num_classes=24
gtarget=0.0
# data_dir='../../../RadioML/Data/GOLD_XYZ_OSC.0001_1024.hdf5'
data_dir="/mnt/delta/Descartes/Git/RadioML_data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

which_gpus="0,1"
epochs=100
batch_size=1024
learning_rate=0.001
decay=1e-5
dataset=RadioML

pretrained_model="saves/2021-09-28_RadioML/RadioML_UltraNet__ep-100_bs-1024_vanilla/model_210928-1756UltraNet.pt"

SUFFIX=vanilla


if [ ! -d "$DIRECTORY" ]; then
    mkdir ./datasets
    mkdir ./saves/${DATE}_${dataset}/
fi

mkdir ./saves/${DATE}_${dataset}/${dataset}_${arch}__ep-${epochs}_bs-${batch_size}_${SUFFIX}

cp train_radioml.sh ./saves/${DATE}_${dataset}/${dataset}_${arch}__ep-${epochs}_bs-${batch_size}_${SUFFIX}/
cp radioml.py ./saves/${DATE}_${dataset}/${dataset}_${arch}__ep-${epochs}_bs-${batch_size}_${SUFFIX}/

$PYTHON radioml.py --data_dir ${data_dir}   \
    --arch ${arch} \
    --save \
    --save_folder ./saves/${DATE}_${dataset}/${dataset}_${arch}__ep-${epochs}_bs-${batch_size}_${SUFFIX} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --decay ${decay} \
    --which_gpus ${which_gpus} \
    --num_classes ${num_classes} \
    --resume ${pretrained_model} \
    # --finetune
    # --test \
    # --finetune \
    #--model_only  --fine_tune\
  
