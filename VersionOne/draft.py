MAZE_S=3
MAZE_T=3
s_=[1,3,3,2,1,1,2,3,2]
block=0
for i in range(MAZE_S):
    for j in range(1,MAZE_T ):
        temp_container = s_[j * MAZE_S + i]
        for k in range(0, j):
            compare_container = s_[k * MAZE_S + i]
            if temp_container < compare_container:
                block = block + 1
print(block)