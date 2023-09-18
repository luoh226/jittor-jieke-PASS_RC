

N_GPU=4
BATCH=96    # 96
FINE_TUNE_BATCH=256   # 256
DATA=data/ImageNetS/ImageNetS50
IMAGENETS=data/ImageNetS/ImageNetS50

DUMP_PATH=weights/res50w2
ARCH=resnet50w2   # resnet18
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
NUM_CLASSES=50
EPOCH=200
EPOCH_PIXELATT=30
EPOCH_SEG=20
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0


#python inference_visualize.py -a ${ARCH} \
#--pretrained ${DUMP_PATH_SEG}/res50w2-4gpus-ckp0-fine20-batch200-28.35_checkpoint.pth.tar \
#--data_path ${IMAGENETS} \
#--dump_path ${DUMP_PATH_SEG} \
#-c ${NUM_CLASSES} \
#--mode validation \
#--match_file ${DUMP_PATH_SEG}/validation/match.json

python inference.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation

#python inference.py -a ${ARCH} \
#--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
#--data_path ${IMAGENETS} \
#--dump_path ${DUMP_PATH_SEG} \
#-c ${NUM_CLASSES} \
#--mode test \
#--match_file ${DUMP_PATH_SEG}/validation/match.json
