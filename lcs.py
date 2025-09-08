def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to get LCS string
    lcs_sequence = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs_sequence.append(seq1[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs_sequence = "".join(reversed(lcs_sequence))
    return dp, lcs_sequence, dp[m][n]


if __name__ == "__main__":
    seq1 = input("Enter Sequence 1: ")
    seq2 = input("Enter Sequence 2: ")

    dp_table, lcs, lcs_length = longest_common_subsequence(seq1, seq2)

    print("\n--- Longest Common Subsequence Results ---")
    print("\nDP Table:")
    print("   " + "  ".join(seq2))
    for i, row in enumerate(dp_table):
        if i == 0:
            print("  ", row)
        else:
            print(seq1[i - 1], row)

    print(f"\nLCS: {lcs}")
    print(f"Length: {lcs_length}")

    if max(len(seq1), len(seq2)) > 0:
        similarity = (lcs_length / max(len(seq1), len(seq2))) * 100
    else:
        similarity = 0
    print(f"Similarity: {similarity:.2f}%")
