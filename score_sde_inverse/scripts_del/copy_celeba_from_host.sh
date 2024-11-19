CONTAINER_ID=$1

if [ -z "$CONTAINER_ID" ]
then
    echo "Please provide a container id"
    exit 1
fi

files=(img_align_celeba.zip list_attr_celeba.txt identity_CelebA.txt list_bbox_celeba.txt list_eval_partition.txt list_landmarks_align_celeba.txt)

for file in "${files[@]}"
do
    docker cp ~/Downloads/$file $CONTAINER_ID:/home/dockeruser/.cache/torchvision/celeba/$file
done
