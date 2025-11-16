BASE=/mnt/data1/syjintw/NEU/mesh-splat/output/main_nerfsynthetic

SCENES=("lego" "ficus" "hotdog" "mic" "ship")

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