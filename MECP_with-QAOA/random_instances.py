info_dim6 = {1: {'exact_covers': ['011110', '101111'],
     'mec': '011110',
     'subsets': [{8, 11},
                 {8, 11, 4, 12},
                 {10, 3},
                 {9, 2, 5, 7},
                 {1, 6},
                 {4, 12}]},
 2: {'exact_covers': ['001101', '111011'],
     'mec': '001101',
     'subsets': [{1, 5},
                 {4, 6},
                 {10, 7},
                 {1, 2, 4, 5, 6, 12},
                 {2, 12},
                 {8, 9, 3, 11}]},
 3: {'exact_covers': ['111100', '000011'],
     'mec': '000011',
     'subsets': [{1, 2},
                 {3, 4, 5, 6},
                 {7, 8, 9},
                 {10, 11, 12},
                 {3, 4, 5, 6, 7},
                 {1, 2, 8, 9, 10, 11, 12}]},
 4: {'exact_covers': ['101011', '100011'],
     'mec': '100011',
     'subsets': [{1, 2},
                 {2, 3},
                 {3, 4},
                 {4, 5},
                 {5, 6, 7, 8},
                 {9, 10, 11, 12}]},
 5: {'exact_covers': ['110001', '011111'],
     'mec': '110001',
     'subsets': [{1, 3, 5, 7, 8, 10},
                 {11, 4},
                 {5, 7},
                 {8, 3},
                 {1, 10},
                 {9, 2, 12, 6}]},
 6: {'exact_covers': ['001111', '111101'],
     'mec': '001111',
     'subsets': [{2, 10},
                 {9, 11, 4, 5},
                 {3, 12},
                 {1, 6},
                 {2, 4, 5, 9, 10, 11},
                 {8, 7}]},
 7: {'exact_covers': ['101001', '011111'],
     'mec': '101001',
     'subsets': [{1, 2, 7, 10, 11, 12},
                 {10, 12},
                 {3, 4},
                 {1, 2},
                 {11, 7},
                 {8, 9, 5, 6}]},
 8: {'exact_covers': ['111100', '011111'],
     'mec': '111100',
     'subsets': [{1, 11, 9, 7},
                 {2, 12},
                 {8, 4},
                 {10, 3, 5, 6},
                 {1, 7},
                 {9, 11}]},
 9: {'exact_covers': ['100111', '111110'],
     'mec': '100111',
     'subsets': [{8, 9, 11, 1},
                 {3, 6},
                 {10, 5},
                 {12, 4},
                 {2, 7},
                 {10, 3, 5, 6}]},
 10: {'exact_covers': ['110010', '111101'],
      'mec': '110010',
      'subsets': [{3, 11, 5, 6},
                  {1, 10},
                  {9, 7},
                  {2, 12},
                  {2, 4, 7, 8, 9, 12},
                  {8, 4}]},
 'U': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}

if __name__ == "__main__":
    import random
    from itertools import combinations
    import pprint

    def is_exact_cover(subsets, bitmask, universe_size=12):
        covered = set()
        for i, bit in enumerate(bitmask):
            if bit == '1':
                covered |= subsets[i]
        return covered == set(range(1, universe_size+1)) and all(
            len(subsets[i] & subsets[j]) == 0
            for i, j in combinations([idx for idx, b in enumerate(bitmask) if b == '1'], 2)
        )

    def find_all_exact_covers(subsets, universe_size=12):
        n = len(subsets)
        exact_covers = []
        for size in range(1, n+1):
            for comb in combinations(range(n), size):
                bitmask = ['0'] * n
                for i in comb:
                    bitmask[i] = '1'
                bitmask_str = ''.join(bitmask)
                if is_exact_cover(subsets, bitmask_str, universe_size):
                    exact_covers.append(bitmask_str)
        return exact_covers

    def generate_instance_with_multiple_exact_covers():
        universe = list(range(1, 13))
        random.shuffle(universe)

        chunk_size = len(universe) // 6
        mec_subsets = []
        for i in range(6):
            if i == 5:
                chunk = universe[i*chunk_size:]
            else:
                chunk = universe[i*chunk_size:(i+1)*chunk_size]
            mec_subsets.append(set(chunk))

        all_subsets = mec_subsets[:]

        idx1, idx2 = random.sample(range(6), 2)
        union_subset = mec_subsets[idx1] | mec_subsets[idx2]
        new_subsets = [all_subsets[i] for i in range(6) if i not in (idx1, idx2)]
        new_subsets.append(union_subset)
        all_subsets = new_subsets

        while len(all_subsets) < 6:
            size = random.randint(1, 6)
            candidate = set(random.sample(universe, size))
            if candidate not in all_subsets:
                all_subsets.append(candidate)

        random.shuffle(all_subsets)

        exact_covers = find_all_exact_covers(all_subsets)

        mec = min(exact_covers, key=lambda x: x.count('1')) if exact_covers else None

        if mec is None or len(exact_covers) < 2:
            return None

        return {
            'subsets': all_subsets,
            'exact_covers': exact_covers,
            'mec': mec
        }

    def generate_info_dim6(n):
        info_dim6 = {'U': set(range(1, 13))}
        count = 1
        while count <= n:
            instance = generate_instance_with_multiple_exact_covers()
            if instance:
                info_dim6[count] = instance
                count += 1
        return info_dim6

    # Esempio per generare 2 istanze nel dizionario info_dim6
    info_dim6 = generate_info_dim6(10)
    pprint.pprint(info_dim6)