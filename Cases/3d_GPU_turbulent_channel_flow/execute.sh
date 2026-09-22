#mpirun -np 4 --hostfile my-hostfile ./RHEA.exe configuration_file.yaml
mpirun --mca btl_base_warn_component_unused 0 -np 4 nsys profile -o timeline_rank%q{OMPI_COMM_WORLD_RANK} --backtrace dwarf  --trace osrt,mpi,nvtx,openacc,cuda --mpi-impl openmpi --sample cpu --cpuctxsw process-tree --cuda-memory-usage true  --stat true --force-overwrite true  ./BSCH_3D_GPU.exe configuration_file.yaml
