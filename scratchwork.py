def visualize_pid_mapping(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    def cdiv(a, b):
        """Integer ceiling division"""
        return -(a // -b)
    
    # Calculate key values
    num_pid_m = cdiv(M, BLOCK_SIZE_M)  # Number of blocks in M dimension
    num_pid_n = cdiv(N, BLOCK_SIZE_N)  # Number of blocks in N dimension
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_pids = num_pid_m * num_pid_n
    
    # Create a matrix to visualize the mapping
    matrix = [[None for _ in range(num_pid_n)] for _ in range(num_pid_m)]
    
    print(f"Matrix dimensions: {M}x{N}")
    print(f"Block sizes: {BLOCK_SIZE_M}x{BLOCK_SIZE_N}")
    print(f"Group size M: {GROUP_SIZE_M}")
    print(f"Number of blocks: {num_pid_m}x{num_pid_n}")
    print("\nPID mapping to (pid_m, pid_n):")
    
    # Calculate mapping for each program ID
    for pid in range(total_pids):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        if pid_m < num_pid_m and pid_n < num_pid_n:
            matrix[pid_m][pid_n] = pid
            print(f"PID {pid:2d} → (m={pid_m}, n={pid_n})")
    
    print("\nMatrix visualization (showing PID for each block):")
    for row in matrix:
        print([f"{x:2d}" if x is not None else "  " for x in row])

# Example with 8x8 matrix, 2x2 blocks, and group size of 2
print("Example with 8x8 matrix:")
visualize_pid_mapping(M=8, N=8, BLOCK_SIZE_M=2, BLOCK_SIZE_N=2, GROUP_SIZE_M=2)


