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

def get_score(a, b, score_matrix):
    if a == "*" or b == "*":
        return -99999  # padding
    # simple cost matrix
    if a == b: return 10
    else     : return -1
    # simple cost matrix
    if (a, b) in score_matrix:
        return score_matrix[(a, b)]
    elif (b, a) in score_matrix:
        return score_matrix[(b, a)]
    return -1  # default

def get_dp(dp, i, j):
    if i < 0 or j < 0:
        return -99999
    return dp[i][j]

def pairwise_alignment(X, Y, score_matrix, gap_penalty=-2):
    X = "*" + X
    Y = "*" + Y
    len_x, len_y = len(X), len(Y)

    dp = [[0] * len_y for _ in range(len_x)]
    traceback = [[None] * len_y for _ in range(len_x)]  # track directions

    # initialize first row/col
    for i in range(1, len_x):
        dp[i][0] = i * gap_penalty
        traceback[i][0] = "U"  # up
    for j in range(1, len_y):
        dp[0][j] = j * gap_penalty
        traceback[0][j] = "L"  # left

    # fill dp table
    for i in range(1, len_x):
        for j in range(1, len_y):
            match = dp[i-1][j-1] + get_score(X[i], Y[j], score_matrix)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty

            dp[i][j] = max(match, delete, insert)

            if dp[i][j] == match:
                traceback[i][j] = "D"  # diagonal
            elif dp[i][j] == delete:
                traceback[i][j] = "U"  # up
            else:
                traceback[i][j] = "L"  # left

    # traceback step
    i, j = len_x - 1, len_y - 1
    align_X, align_Y = [], []

    while i > 0 or j > 0:
        if traceback[i][j] == "D":
            align_X.append(X[i])
            align_Y.append(Y[j])
            i -= 1
            j -= 1
        elif traceback[i][j] == "U":
            align_X.append(X[i])
            align_Y.append("-")
            i -= 1
        elif traceback[i][j] == "L":
            align_X.append("-")
            align_Y.append(Y[j])
            j -= 1
        else:
            break

    align_X.reverse()
    align_Y.reverse()

    return dp, "".join(align_X), "".join(align_Y)

def pa_pair(x, y):
    score_matrix = load_blosum62("blosum62.txt")
    dp, aligned_x, aligned_y = pairwise_alignment(x, y, score_matrix)
    print("DP matrix:\n", np.array(dp))
    print("\nAligned sequences:")
    print(aligned_x)
    print(aligned_y)

if __name__ == "__main__":
    x1 = "GGATTGT"
    x2 = "GGAAGG"
    x3 = "AAGGTT"
    x4 = "AGGT"
    pa_pair(x1, x3)
    pa_pair(x1, x4)
    pa_pair(x2, x3)
    pa_pair(x2, x4)


    # l = [x1, x2, x3, x4]
    # for i in range(len(l)):
    #     for j in range(len(l)):
    #         if i >= j: continue
    #         print(f"pair{i+1}, {j+1}")
    #         pa_pair(l[i], l[j])
