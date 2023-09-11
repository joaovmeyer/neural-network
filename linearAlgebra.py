def dot(v1, v2):
	ans = 0;

	for i in range(len(v1)):
		ans += v1[i] * v2[i];

	return ans;


def transpose(mat):
    ans = [];
    for i in range(len(mat[0])):
        ans.append([]);
        
        for j in range(len(mat)):
            ans[i].append(mat[j][i]);
            
    return ans;


def matMultiplyVector(mat, vec):

	ans = [];

	for i in range(len(mat)):
		ans.append(0);
		for j in range(len(mat[i])):
			ans[i] += vec[j] * mat[i][j];

	return ans;

def matMultiplyMat(mat1, mat2):
	# if len(mat1[0]) != len(mat2):
	# 	raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

	ans = [];

	for i in range(len(mat1)):
		ans.append([]);
		for j in range(len(mat2[0])):
			ans[i].append(0);

			for k in range(len(mat1[0])):
				ans[i][j] += mat1[i][k] * mat2[k][j];

	return ans;