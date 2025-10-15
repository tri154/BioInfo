import numpy as np
def load_blosum62(filepath):
    matrix = {}
    with open(filepath) as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
        headers = lines[0].split()
        for row in lines[1:]:
            parts = row.split()
            aa = parts[0]
            scores = list(map(int, parts[1:]))
            for h, s in zip(headers, scores):
                matrix[(aa, h)] = s
    return matrix
    # print(blosum62[('A', 'R')])  # should print -1

def get_score(a, b, score_matrix):
    if a == b:
        return 2
    else:
        return -1

    # if a == "*" or b == "*":
    #     return -99999
    # if (a, b) in score_matrix:
    #     return score_matrix[(a, b)]
    # elif (b, a) in score_matrix:
    #     return score_matrix[(b, a)]
    # else:
    #     return None

def get_dp(dp, i, j):
    # len_x, len_y = len(dp), len(dp[0])
    if i < 0 or j < 0:
        return -99999
    return dp[i][j]

def pairwise_aligment(X, Y, score_matrix):
    X = "*" + X
    Y = "*" + Y
    len_x, len_y = len(X), len(Y)
    dp = [[0] * (len_y) for _ in range(len_x)]
    for i in range(0, len_x):
        for j in range(0, len_y):
            if i == 0 and j == 0:
                continue
            dp[i][j] = max(
                get_dp(dp, i-1, j-1) + get_score(X[i], Y[j], score_matrix),
                get_dp(dp, i-1, j) - 2,
                get_dp(dp, i, j-1) - 2,
            )
    return dp


if __name__ == "__main__":
    score_matrix = load_blosum62("blosum62.txt")
    x = "GGATTGT"
    y = "GGAAGG"
    dp = pairwise_aligment(x, y, score_matrix)
    print(np.array(dp))
