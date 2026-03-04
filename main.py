from seq_len_balance import kk, ffd, bfd

print(kk([8, 7, 6, 5, 4], 2))  # -> [[8, 5], [7, 4], [6]]  (approx)
print(ffd([6, 3, 4, 5, 2, 7, 1], 10))  # -> [[7, 3], [6, 4], [5, 2, 1]]
print(bfd([6, 3, 4, 5, 2, 7, 1], 10))  # -> [[7, 2, 1], [6, 4], [5, 3]]
