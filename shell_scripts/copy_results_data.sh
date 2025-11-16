BASE=/mnt/data1/syjintw/NEU/mesh-splat/output/main_nerfsynthetic

SCENES=("lego"  "ficus" "hotdog" "mic" "ship")


for SCENE in ${SCENES[@]}
do
    echo "Copying scene: $SCENE"
    JSON_DIR=${BASE}/${SCENE}/for_plot
    cp --parents -r $JSON_DIR ./data/
done



    
