
BASE=/mnt/data1/syjintw/NEU/mesh-splat/output/main_nerfsynthetic
# BASE=/mnt/data1/syjintw/NEU/mesh-splat-mixed/output/main_nerfsynthetic
SCENES=("lego" "ficus" "hotdog" "mic" "ship")


echo "copying mesh_splat results data to ./data/ ..."

for SCENE in ${SCENES[@]}
do
    echo "Copying per_view_gs_mesh.json files for scene: $SCENE"
    
    # Find all per_view_gs_mesh.json files in the scene directory
    find ${BASE}/${SCENE} -name "per_view_gs_mesh.json" | while read json_file; do
        # Get the relative path from BASE
        rel_path=${json_file#${BASE}/}
        
        # Create destination directory structure
        dest_dir=./data/$(dirname $rel_path)
        mkdir -p $dest_dir
        
        # Copy the file
        cp $json_file $dest_dir/
        echo "  Copied: $rel_path"
    done
done

echo "Done! All per_view_gs_mesh.json files copied to ./data/"


BASE=/mnt/data1/syjintw/NEU/mesh-splat/output/main_mipnerf360

SCENES=("bicycle" "bicycle-dw10" "bicycle-dw30" "bicycle-dw50")

for SCENE in ${SCENES[@]}
do
    echo "Copying per_view_gs_mesh.json files for scene: $SCENE"
    
    # Find all per_view_gs_mesh.json files in the scene directory
    find ${BASE}/${SCENE} -name "per_view_gs_mesh.json" | while read json_file; do
        # Get the relative path from BASE
        rel_path=${json_file#${BASE}/}
        
        # Create destination directory structure
        dest_dir=./data/$(dirname $rel_path)
        mkdir -p $dest_dir
        
        # Copy the file
        cp $json_file $dest_dir/
        echo "  Copied: $rel_path"
    done
done

echo "Done! All per_view_gs_mesh.json files copied to ./data/bicycle_exps/"

echo "\n\n\n"



# ============================================================

GAMES_DATA_DIR="./data/games_results"
echo "Copying GaMeS results data to $GAMES_DATA_DIR/ ..."


BASE=/mnt/data1/syjintw/NEU/mesh-splat-games/output/main_nerfsynthetic
SCENES=("lego" "ficus" "hotdog" "mic" "ship")

for SCENE in ${SCENES[@]}
do
    echo "Copying per_view_gs_mesh.json files for scene: $SCENE"
    
    # Find all per_view_gs_mesh.json files in the scene directory
    find ${BASE}/${SCENE} -name "per_view_gs_mesh.json" | while read json_file; do
        # Get the relative path from BASE
        rel_path=${json_file#${BASE}/}
        
        # Create destination directory structure
        dest_dir=$GAMES_DATA_DIR/$(dirname $rel_path)
        mkdir -p $dest_dir
        
        # Copy the file
        cp $json_file $dest_dir/
        echo "  Copied: $rel_path"
    done
done

echo "Done! All per_view_gs_mesh.json files copied to ./data/"


BASE=/mnt/data1/syjintw/NEU/mesh-splat-games/output/main_mipnerf360
SCENES=("bicycle" "bicycle-dw10" "bicycle-dw30" "bicycle-dw50")

for SCENE in ${SCENES[@]}
do
    echo "Copying per_view_gs_mesh.json files for scene: $SCENE"
    
    # Find all per_view_gs_mesh.json files in the scene directory
    find ${BASE}/${SCENE} -name "per_view_gs_mesh.json" | while read json_file; do
        # Get the relative path from BASE
        rel_path=${json_file#${BASE}/}
        
        # Create destination directory structure
        dest_dir=$GAMES_DATA_DIR/$(dirname $rel_path)
        mkdir -p $dest_dir
        
        # Copy the file
        cp $json_file $dest_dir/
        echo "  Copied: $rel_path"
    done
done

echo "Done ! All per_view_gs_mesh.json files copied to $GAMES_DATA_DIR/"

