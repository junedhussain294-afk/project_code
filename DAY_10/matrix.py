
mat1 = [[1, 2, 3],
        [4, 5, 6]]

mat2 = [[7, 8, 9],
        [10, 11, 12]]


result = []
for i in range(len(mat1)):
    row = []
    for j in range(len(mat1[0])):
        row.append(mat1[i][j] + mat2[i][j])
    result.append(row)


print("Matrix Addition Result:")
for r in result:
    print(r)
