set root_path /its/home/tafj0/python/ETTrack2/datasets/mot17/train/
set out_path processed_mot/
#set -gx PYTORCH_CUDA_ALLOC_CONF expandable_segments:True

for f in $root_path*
	echo $f
	set base_name (path basename $f)
	echo $base_name
	set full_out_path $out_path$base_name
	echo $full_out_path
	mkdir -p $full_out_path
	uv run run.py --input_video $f --output_dir  $full_out_path --encoder vitl --save_npz

end


